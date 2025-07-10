prompt=$1
start_img=$2 

curl --request POST \
  --url https://api.piapi.ai/api/kling/v1/video \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header 'X-API-key: ' \
  --data '{
  "negative_prompt": "",
  "prompt": "'"${prompt}"'",
  "version": "1.6",
  "creativity": 0.5,
  "duration": 5,
  "aspect_ratio": "",
  "professional_mode": true,
  "image_url": "'"${start_img}"'"
  }'

# check https://piapi.ai/docs/kling-api/video-generation for more information on configurations
# PUT YOUR OWN PiAPI X-API-KEY