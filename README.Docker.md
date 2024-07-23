### Building and running your application

When you're ready, build your application by running:
`docker compose build --no-cache`

Then, start the application:
`docker compose up`

You can use postman to send the image for prediction. The will be available at:

[loca](http://127.0.0.1:5000/predict)

The body requires a key-value.

Select he option of "form-data"

The key is the string "image".

The value is the path of the image that you want to predict.

Example:
https://go.postman.co/workspace/9bcc5549-42d7-46be-879d-dbcd9db46905/request/19482413-5439ecfc-7b9f-46d7-9100-5e57d2f225a4
