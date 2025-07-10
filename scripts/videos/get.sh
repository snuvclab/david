id=$1

curl --request GET \
  --url https://api.piapi.ai/api/kling/v1/video/$id \
  --header 'Accept: application/json' \
  --header 'X-API-Key: '

# check https://piapi.ai/docs/kling-api/video-generation for more information on configurations
# PUT YOUR OWN PiAPI X-API-KEY