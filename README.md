# ai services

Contains the ai services for foresight-next:

* [Load Forecasting](./load-forecasting/README.md)
* [Nonintrusive Load Monitoring (NILM)](./nilm/README.md)
* [Wakeup Detection](./wakeup-detection/README.md)

See [Packages](https://github.com/orgs/connected-intelligent-systems/packages?repo_name=foresight-next-ai-services) for the available images.

The build workflow publishes CPU and GPU variants of all three services:

| Variant | Platforms | Example tags |
| --- | --- | --- |
| CPU (default) | `linux/amd64`, `linux/arm64` | `latest`, `main` |
| GPU | `linux/amd64` | `latest-gpu`, `main-gpu` |

GPU variants use an NVIDIA GPU when available. Run them with Docker GPU access enabled, for example:

```bash
docker run --gpus all -p 8000:8000 ghcr.io/connected-intelligent-systems/foresight-next-ai-services/load-forecasting:latest-gpu
```

Wakeup detection uses TensorFlow 2.15.1 to support ARM64 while retaining Keras 2 compatibility with the bundled models.

## Authors

Alexander Anisimov <alexander.anisimov@dfki.de>  
Muhammad Muaz <muhammad.muaz@dfki.de>
