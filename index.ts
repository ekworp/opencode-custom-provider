import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Model, Plugin, Provider } from "@opencode/plugin";

/**
 * Registers custom OpenAI-compatible providers and discovers their models from
 * `GET {baseURL}/models`.
 *
 * Configuration only needs an endpoint and a key. Each provider gets its own
 * provider ID, so built-in providers such as `openai` keep working unchanged.
 */

const DEFAULTS = {
	package: "@opencode/ai/providers/openai-compatible",
	refreshInterval: 6 * 60 * 60,
};

const HOSTNAME_NOISE = new Set([
	"api",
	"www",
	"llm",
	"openai",
	"chat",
	"gateway",
	"proxy",
	"v1",
	"com",
	"net",
	"org",
	"io",
	"ai",
	"dev",
	"cn",
]);

type ProviderSpec = {
	/** Provider prefix used in model references, e.g. `gateway/example-model`. Derived from the host when omitted. */
	id?: string;
	/** Display name. Defaults to `id`. */
	name?: string;
	/** OpenAI-compatible endpoint, including `/v1`. */
	baseURL?: string;
	/** Inline API key. Prefer `apiKeyEnv` when possible. */
	apiKey?: string;
	/** Environment variable holding the API key. Read before `apiKey`. */
	apiKeyEnv?: string;
	/** Runtime package. Defaults to the OpenAI-compatible package. */
	package?: string;
	/** Extra HTTP headers sent to both discovery and model requests. */
	headers?: Record<string, string>;
	/** Local model metadata overrides. Only configured fields are applied. */
	models?: Record<string, ModelOptions>;
	/** Reasoning variants to expose for every discovered model. */
	reasoning?: ReasoningOptions;
};

type ModelOptions = {
	/** Context window in tokens. */
	context?: number;
	/** Maximum output tokens. */
	output?: number;
};

type ReasoningOptions = {
	/** Effort values exposed as model variants, for example `low`, `medium`, and `high`. */
	efforts?: string[];
	/** Request option name used by the upstream endpoint. Defaults to `reasoningEffort`. */
	parameter?: string;
	/** Also expose a `none` variant that disables reasoning. */
	none?: boolean;
	/** Response field containing streamed reasoning, when the endpoint uses a custom field. */
	field?: string;
};

type Options = ProviderSpec & {
	/** Define several providers with one plugin instance. */
	providers?: ProviderSpec[];
	/** Refresh interval in seconds. `0` disables refreshing. */
	refreshInterval?: number;
};

type ModelsResponse = {
	data?: Array<{ id?: string; label?: string }>;
};

type ResolvedSpec = Required<
	Pick<ProviderSpec, "id" | "name" | "baseURL" | "package">
> &
	Pick<ProviderSpec, "headers" | "models" | "reasoning"> & {
		apiKey: string;
	};

type Source = {
	spec: ResolvedSpec;
	models: Model.Info[];
};

type RefreshSubscriber = {
	refresh: () => Promise<boolean>;
	interval: number;
	nextAt: number;
};

// OpenCode may create several plugin instances for different locations in the
// same process. Keep one process-wide scheduler instead of one timer per
// location. Each location still owns its provider state and refresh callback.
const refreshSubscribers = new Set<RefreshSubscriber>();
let refreshTimer: ReturnType<typeof setTimeout> | undefined;

type ModelRequestCacheEntry = {
	models?: Model.Info[];
	promise?: Promise<Model.Info[]>;
	completedAt: number;
};

// Several locations can initialize at nearly the same time. Coalesce their
// discovery requests and briefly reuse the completed result as well. The cache
// key includes the request configuration so different providers never share a
// response accidentally.
const MODEL_REQUEST_CACHE_TTL = 60_000;
const modelRequestCache = new Map<string, ModelRequestCacheEntry>();
const sharedCacheDirectory = join(tmpdir(), "opencode", "custom-provider");

function modelRequestKey(spec: ResolvedSpec): string {
	return JSON.stringify({
		baseURL: spec.baseURL,
		apiKey: spec.apiKey,
		headers: spec.headers,
		models: spec.models,
		reasoning: spec.reasoning,
	});
}

function sharedCachePaths(key: string): { cache: string; lock: string } {
	const digest = createHash("sha256").update(key).digest("hex");
	return {
		cache: join(sharedCacheDirectory, `${digest}.json`),
		lock: join(sharedCacheDirectory, `${digest}.lock`),
	};
}

async function readSharedModels(
	path: string,
): Promise<Model.Info[] | undefined> {
	try {
		const value = JSON.parse(await readFile(path, "utf8")) as {
			completedAt?: number;
			models?: Model.Info[];
		};
		if (
			typeof value.completedAt === "number" &&
			Array.isArray(value.models) &&
			Date.now() - value.completedAt < MODEL_REQUEST_CACHE_TTL
		)
			return value.models;
	} catch {
		// The cache is best effort. Discovery still works if it is unavailable.
	}
	return undefined;
}

const sleep = (milliseconds: number) =>
	new Promise<void>((resolve) => setTimeout(resolve, milliseconds));

function scheduleRefreshTimer(): void {
	if (refreshTimer !== undefined) {
		clearTimeout(refreshTimer);
		refreshTimer = undefined;
	}
	if (refreshSubscribers.size === 0) return;

	const nextAt = Math.min(
		...Array.from(refreshSubscribers, (subscriber) => subscriber.nextAt),
	);
	const delay = Math.max(0, nextAt - Date.now());
	refreshTimer = setTimeout(async () => {
		refreshTimer = undefined;
		const now = Date.now();
		const due = Array.from(refreshSubscribers).filter(
			(subscriber) => subscriber.nextAt <= now,
		);

		for (const subscriber of due) {
			subscriber.nextAt = now + subscriber.interval;
			try {
				await subscriber.refresh();
			} catch (error) {
				console.error(
					`[custom-provider] scheduled refresh failed: ${error instanceof Error ? error.message : error}`,
				);
			}
		}

		scheduleRefreshTimer();
	}, delay);
	refreshTimer.unref?.();
}

function subscribeToRefresh(
	refresh: () => Promise<boolean>,
	interval: number,
): () => void {
	const subscriber: RefreshSubscriber = {
		refresh,
		interval,
		nextAt: Date.now() + interval,
	};
	refreshSubscribers.add(subscriber);
	scheduleRefreshTimer();

	return () => {
		refreshSubscribers.delete(subscriber);
		scheduleRefreshTimer();
	};
}

/** `https://api.acme.example/v1` -> `acme`, so `id` can usually be omitted. */
function deriveId(baseURL: string): string {
	try {
		const host = new URL(baseURL).hostname;
		if (host === "localhost" || /^\d+(\.\d+){3}$/.test(host)) return "local";
		const label = host
			.split(".")
			.find((part) => part && !HOSTNAME_NOISE.has(part));
		return (label ?? "custom").replace(/[^a-zA-Z0-9_-]/g, "") || "custom";
	} catch {
		return "custom";
	}
}

function resolveSpec(
	raw: ProviderSpec,
	apiKeyEnvFallback?: string,
): ResolvedSpec | undefined {
	const baseURL = String(raw.baseURL ?? "").trim();
	if (!baseURL) {
		console.error(
			"[custom-provider] skipping provider entry without `baseURL`",
		);
		return undefined;
	}

	const id = String(raw.id ?? deriveId(baseURL)).trim();
	if (!id) {
		console.error(
			`[custom-provider] could not derive a provider id for ${baseURL}, set options.id`,
		);
		return undefined;
	}
	if (id.includes("/") || id.includes("#")) {
		console.error(
			`[custom-provider] invalid provider id "${id}": must not contain "/" or "#"`,
		);
		return undefined;
	}

	const envName = raw.apiKeyEnv ?? apiKeyEnvFallback;
	const fromEnv = envName ? process.env[envName] : undefined;
	const apiKey = fromEnv ?? raw.apiKey ?? "";

	return {
		id,
		name: raw.name ?? id,
		baseURL: baseURL.replace(/\/+$/, ""),
		package: raw.package ?? DEFAULTS.package,
		headers: raw.headers,
		models: raw.models,
		reasoning: raw.reasoning,
		apiKey,
	};
}

async function loadModels(spec: ResolvedSpec): Promise<Model.Info[]> {
	const headers: Record<string, string> = { ...spec.headers };
	if (spec.apiKey) headers.Authorization = `Bearer ${spec.apiKey}`;

	const res = await fetch(`${spec.baseURL}/models`, {
		headers,
		signal: AbortSignal.timeout(30_000),
	});
	if (!res.ok)
		throw new Error(`GET ${spec.baseURL}/models failed: HTTP ${res.status}`);

	const body = (await res.json()) as ModelsResponse;
	if (!Array.isArray(body.data))
		throw new Error("response has no `data` array");

	const providerID = Provider.ID.make(spec.id);
	const entries = body.data
		.map((model) => ({
			id: model?.id?.trim(),
			label: model?.label?.trim(),
		}))
		.filter((model): model is { id: string; label: string | undefined } =>
			Boolean(model.id),
		);

	return entries.map(({ id, label }) => {
		const options = spec.models?.[id];
		const defaults = Model.Info.default(providerID, Model.ID.make(id));
		const model: Model.Info = {
			...defaults,
			name: label || id,
			...(options?.context !== undefined || options?.output !== undefined
				? {
						limit: {
							...defaults.limit,
							...(options.context !== undefined
								? { context: options.context }
								: {}),
							...(options.output !== undefined
								? { output: options.output }
								: {}),
						},
					}
				: {}),
		};
		const reasoning = spec.reasoning;
		if (!reasoning) return model;

		const parameter = reasoning.parameter ?? "reasoningEffort";
		const efforts = [...new Set(reasoning.efforts ?? ["low", "medium", "high"])]
			.map((effort) => effort.trim())
			.filter(Boolean);
		const variants = efforts.map((effort) => ({
			id: Model.VariantID.make(effort),
			settings: { [parameter]: effort },
		}));
		if (reasoning.none)
			variants.unshift({
				id: Model.VariantID.make("none"),
				settings: { [parameter]: "none" },
			});

		return {
			...model,
			variants,
			...(reasoning.field
				? {
						compatibility: {
							...model.compatibility,
							reasoningField: reasoning.field,
						},
					}
				: {}),
		};
	});
}

async function loadModelsDeduplicated(
	spec: ResolvedSpec,
	options: { force?: boolean } = {},
): Promise<{ models: Model.Info[]; fetched: boolean }> {
	const key = modelRequestKey(spec);
	const now = Date.now();
	const cached = modelRequestCache.get(key);
	if (cached?.promise) return { models: await cached.promise, fetched: false };
	if (
		!options.force &&
		cached?.models &&
		now - cached.completedAt < MODEL_REQUEST_CACHE_TTL
	)
		return { models: cached.models, fetched: false };

	const paths = sharedCachePaths(key);
	await mkdir(sharedCacheDirectory, { recursive: true });

	// The lock file is shared by separate OpenCode location/VM contexts.
	// Whichever context acquires it performs the request; the others wait for
	// the resulting cache file instead of sending duplicate requests.
	for (;;) {
		if (!options.force) {
			const shared = await readSharedModels(paths.cache);
			if (shared) return { models: shared, fetched: false };
		}

		let ownsLock = false;
		try {
			await writeFile(paths.lock, String(process.pid), { flag: "wx" });
			ownsLock = true;
		} catch {
			try {
				const lockAge = Date.now() - (await stat(paths.lock)).mtimeMs;
				if (lockAge > 35_000) await rm(paths.lock, { force: true });
			} catch {
				// The lock may have disappeared between stat and removal.
			}
			await sleep(50);
			continue;
		}

		if (!ownsLock) continue;
		const entry: ModelRequestCacheEntry = { completedAt: 0 };
		const promise = loadModels(spec)
			.then(async (models) => {
				entry.models = models;
				entry.completedAt = Date.now();
				const temporary = `${paths.cache}.${process.pid}.${Date.now()}.tmp`;
				await writeFile(
					temporary,
					JSON.stringify({ completedAt: entry.completedAt, models }),
					"utf8",
				);
				await rename(temporary, paths.cache);
				return models;
			})
			.catch((error) => {
				if (modelRequestCache.get(key) === entry) modelRequestCache.delete(key);
				throw error;
			})
			.finally(() => rm(paths.lock, { force: true }));
		entry.promise = promise;
		modelRequestCache.set(key, entry);
		return { models: await promise, fetched: true };
	}
}

export default Plugin.define({
	id: "custom-provider.models",
	async setup(ctx) {
		const options = ctx.options as Options;
		const rawSpecs = Array.isArray(options.providers)
			? options.providers
			: [options];

		const sources = new Map<string, Source>();
		for (const raw of rawSpecs) {
			const spec = resolveSpec(raw, options.apiKeyEnv);
			if (!spec) continue;
			if (sources.has(spec.id)) {
				console.error(
					`[custom-provider] duplicate provider id "${spec.id}", set an explicit options.id`,
				);
				continue;
			}

			try {
				const { models, fetched } = await loadModelsDeduplicated(spec);
				sources.set(spec.id, { spec, models });
				if (fetched)
					console.log(
						`[custom-provider] ${spec.id}: discovered ${models.length} models from ${spec.baseURL}/models`,
					);
			} catch (error) {
				console.error(
					`[custom-provider] ${spec.id}: discovery failed: ${error instanceof Error ? error.message : error}`,
				);
			}
		}

		if (sources.size === 0) return;

		await ctx.provider.transform((editor) => {
			for (const { spec, models } of sources.values()) {
				editor.add({
					info: {
						...Provider.Info.empty(Provider.ID.make(spec.id)),
						name: spec.name,
						activation: "enabled",
						package: spec.package,
						settings: { baseURL: spec.baseURL, apiKey: spec.apiKey },
						...(spec.headers ? { headers: spec.headers } : {}),
					},
					models,
				});
			}
		});

		const refresh = async (force = false) => {
			let changed = false;
			for (const source of sources.values()) {
				try {
					const { models, fetched } = await loadModelsDeduplicated(
						source.spec,
						{ force },
					);
					source.models = models;
					changed = true;
					if (fetched)
						console.log(
							`[custom-provider] ${source.spec.id}: refreshed ${models.length} models from ${source.spec.baseURL}/models`,
						);
				} catch (error) {
					console.error(
						`[custom-provider] ${source.spec.id}: refresh failed: ${error instanceof Error ? error.message : error}`,
					);
				}
			}
			if (changed) await ctx.provider.reload();
			return changed;
		};

		await ctx.command.transform((editor) => {
			editor.add({
				name: "custom-provider-refresh",
				description: "Refresh models from all custom provider endpoints now",
				execute: async () => {
					await refresh(true);
				},
			});
		});

		const interval =
			Number(options.refreshInterval ?? DEFAULTS.refreshInterval) * 1000;
		if (!Number.isFinite(interval) || interval <= 0) return;

		return subscribeToRefresh(refresh, interval);
	},
});
