from utils.ainaz.imageLoader import load_image

with open("utils/ainaz/test.jpg", "rb") as f:
    file_bytes = f.read()

class FakeUpload:
    def __init__(self, file_bytes):
        self.file_bytes = file_bytes
        self.name = "test.jpg"
        self.type = "image/jpeg"

    def read(self):
        return self.file_bytes

fake_file = FakeUpload(file_bytes)

img_data = load_image(fake_file)

print("Имя:", img_data.filename)
print("Размер:", img_data.width, "x", img_data.height)
print("Формат:", img_data.image_format)
print("Файл (bytes):", img_data.file_size)
print("Shape:", img_data.image.shape)