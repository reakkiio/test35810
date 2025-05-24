from webscout.Provider.TTI import PixelMuse
client = PixelMuse()
response = client.images.create(
    model="flux-schnell",
    prompt="a white siamese cat",
    response_format="url",
    n=4,
    timeout=30,
    image_format="png"
)
print([d.url for d in response.data])
