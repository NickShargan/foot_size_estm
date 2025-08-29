# Foot size visual estimation

## Usage

Run: ```python foot_size_estm.py```

It should return:

```
image 1/1 ./foot_size_estm/IMG_7350.jpg: 1024x1024 1 0, 1 1, 1 2, 1 3, 1 4, 1 5, 1 6, 4856.7ms
Speed: 12.2ms preprocess, 4856.7ms inference, 3.3ms postprocess per image at shape (1, 3, 1024, 1024)
215.9
Distance between projected 'most_left' and 'most_right'           (mm): 106.48
Distance between projected points (pixels): 1001.02

foot length: 272.10 mm
foot width: 106.48 mm
foot size: US 9.0; EU 42.5;          UK 8.5
foot width: 2E

image with size visualisations is stored at:          ./foot_size_estm/IMG_7350_vis.jpg
```

## Example

![Alt text](foot_size_estm/IMG_7350.jpg)


## Build & Deploy
```
<!-- IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/foot-size-api:v3" -->

PROJECT_ID=$(gcloud config get-value project)
REGION=northamerica-northeast1
REPO=containers
SERVICE=foot-size-api
TAG=$(date +%Y%m%d%H%M)         # unique tag; or use: $(git rev-parse --short HEAD)

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:$TAG"

gcloud builds submit --tag "$IMAGE" .
gcloud run deploy foot-size-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi --concurrency 1 --timeout 600
```

```
curl -X POST "https://foot-size-api-762504128529.northamerica-northeast1.run.app/measure"   -F "file=@./foot_size_estm/IMG_7350.jpg"   -F "gender=m" -F "ref_obj=paper_letter" -F "is_wall=true" -F "return_vis=false"  | jq .
```

