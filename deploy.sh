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
  --cpu 2 --memory 4Gi --concurrency 1 --timeout 600

curl -X POST "https://foot-size-api-762504128529.northamerica-northeast1.run.app/measure"   -F "file=@./foot_size_estm/IMG_7350.jpg"   -F "gender=m" -F "ref_obj=paper_letter" -F "is_wall=true" -F "return_vis=false"  | jq .