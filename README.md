# ESCA AI Inference Service

One service runs the board's models for everything else: an application
asks for a model by name and never links a framework itself.

    application ─► libesca-inference-client ─D-Bus─► esca-inferenced ─► TFLite
                         │                                  │
                         └──── tensors in shared memory ────┘

## Shape

| Part | Does |
| --- | --- |
| Model registry | Reads one JSON configuration per model from `models.d/`; the name must match the model file's name |
| Model manager | Loads a model on its first request (or at start with `preload`), unloads idle ones under memory pressure |
| Scheduler | Sends each request to a free backend, falling back to the next one the model allows; swappable scheduling and eviction policies |
| Engine | One model on one backend; TFLite with XNNPACK or CPU in this build |
| Context manager | A context per client: input shapes and a shared-memory buffer per tensor, so pixels never cross D-Bus; it dies with its client |
| Capability | The models it can run are registered with esca-capabilityd as the `ai` section |

## A model

    {
        "name": "detect",
        "modelPath": "/usr/share/esca/models/detect.tflite",
        "backend": ["XNNPACK", "CPU"],
        "threads": 3,
        "preload": false
    }

## D-Bus: `vn.sparc.esca.Inference1`

Models: `ListModels`, `GetModelConfig`, `GetModelInfo`, `RegisterModel`,
`UnregisterModel`, `LoadModel`, `UnloadModel`, `Refresh`.
Contexts: `CreateContext(Async)`, `GetContext`, `DestroyContext`,
`SetInputShape`. Running: `Infer`, `InferAsync`, `GetStats`,
`SetSchedulingPolicy`, `SetEvictionPolicy`. Signals include
`ModelStateChanged` and `ContextCreated`.

The client library offers `InferenceBuilder`, which runs a model either in
the service or in the application's own process (`local` mode) behind the
same `Session`.

## From a terminal

    esca-inferctl list
    esca-inferctl info detect
    esca-inferctl run detect --repeat 20
    esca-inferctl stats

## Files

`/etc/esca/inference.conf`, the model configurations, the systemd unit and
the bus policy are in `data/`. Built by the `esca-inference` recipe in
meta-sparc-esca.
