# Using the load forecasting service

You can run the service via the provided image.
The container is exposing port `8000`.
```
docker run -p 8000:8000 ghcr.io/connected-intelligent-systems/foresight-next-ai-services/wakeup-detection:latest
```

In this example, it is forwarded to the same port of the local host.
The service is now running and can be interacted with
via synchronous `POST` requests to port `8000` of the machine hosting the container
or any other port the container is forwarding to.
The response contains the model estimations.

You can check whether everything works as intended by sending a test request.
To this end, clone the code then `cd` to the service directory
(the directory containing this README) and execute the following command:

```
curl http://127.0.0.1:8000/estimate --request POST --header "Content-Type: application/json" --data @./res/sample_request.json
```

## Expected input format

JSON formatted like [sample_request.json](./res/sample_request.json)

- POST request with a header: `Content-Type: application/json`
- Time as string in ISO format: e.g. `"2020-11-01T01:05:19+01:00"`
- Power in Watt
- A time window of at least 90 minutes 
  (referring to the distance between the first and the last measurement)
- A resolution that can reasonably resampled to minute-wise
  (avoid gaps longer than several minutes)

## Output format

JSON file containing the following fields:

- `confid_asleep`: `float`,
  the confidence score for the wake-up event having not yet occurred
- `confid_recent_wu`: `float`,
  the confidence score for the wake-up event having occurred during the provided time window
- `confid_awake`: `float`,
  the confidence score for the wake-up event having occurred before the provided time window
- `wu_time`: `str`,
  the time stamp in ISO format for the estimated time of the wake-up event.
  This is only relevant if `confid_recent_wu` is above 0.97.

e.g.
```
{"confid_asleep": 0.9678203463554382, "confid_recent_wu": 0.030633417889475822, "confid_awake": 0.001546248677186668, "wu_time": "2020-11-01 02:29:00+01:00"}
```
