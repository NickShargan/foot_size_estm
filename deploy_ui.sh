PROJECT_ID=$(gcloud config get-value project)
REGION=northamerica-northeast1
REPO=containers
SERVICE_UI=foot-ui
TAG=$(date +%Y%m%d%H%M)

IMAGE_UI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE_UI:$TAG"

gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE_UI="$IMAGE_UI" .

gcloud run deploy $SERVICE_UI \
  --image "$IMAGE_UI" \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated
