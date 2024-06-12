
[//]: # (```markdown)
# ForeSightNEXT NILM Service

This project provides a Flask application that receives energy data from the smart meter as input and outputs the disaggregation of the energy into appliances.

## Getting Started

These instructions will guide you on how to run the app using Docker.

### Prerequisites

- Docker installed on your machine
- Git installed on your machine

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/connected-intelligent-systems/foresight-next-ai-services.git

### Build the Docker Image
### Navigate to the nilm directory containing the Dockerfile and build the Docker image:

cd nilm
docker build -t ${image_name} .
```

Replace `${image_name}` with your desired image name. 

### Run the Docker Container

To run the container, use the following command:

```bash
docker run -d -e APPLIANCE=${appliance_name} -e hid=${house_id} -p <host_port>:8000 --name=${container_name} ${image_name}
```

In this command:
- `${appliance_name}` is the short form of the appliance type. The current possible appliance types and their short forms are:
  - "DW": 'Dishwasher'
  - "FRDFZ": 'Fridge-Freezer'
  - "KTL": 'Kettle'
  - "MW": 'Microwave'
  - "TV": 'TV'
  - "WM": 'Washing Machine'
- `${house_id}` is the house ID. The current possible house IDs are 5 and 9.
- `<host_port>` is the port on your host machine that you want to map to port 8000 of the container.
- `${container_name}` is your desired container name.
- `${image_name}` is your desired image name.

For example, to run a container for house 5 for a dishwasher at port 9000, use the following command:

```bash
docker run -d -e APPLIANCE=DW -e hid=5 -p 9000:8000 --name=dw_container fsnext/nilm_sample
```

The container accepts POST requests at the `/predict` endpoint (http://localhost:<host_port>/predict).

### Input Format

The input should be a JSON array of objects, where each object has a 'time' field and a 'value' field. The 'time' field is the timestamp of the sample in the format "YYYY-MM-DDTHH:mm:ss.sssZ", and the 'value' field is the total power drawn recorded by the smart meter.

Example:

```json
[
    {
        "time": "2024-04-02T07:16:58.016Z",
        "value": 326
    },
    {
        "time": "2024-04-02T07:16:58.030Z",
        "value": 326
    },
    ...
]
```

The endpoint returns the disaggregation of the appliance in a similar format.

### Example Code

Example code to access the container is present in the following notebook: `nilm/src/example_test.ipynb`.