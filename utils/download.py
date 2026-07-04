import os
import requests

PUBLIC_KEY = "https://disk.yandex.ru/d/0Gf_6d61RiQsTQ"
FILENAME = "best_model_talc.pth"
SAVE_PATH = "artifacts/checkpoints"


def download():
    os.makedirs(SAVE_PATH, exist_ok=True)

    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

    r = requests.get(api_url, params={"public_key": PUBLIC_KEY})
    
    # ВАЖНО: сначала смотри ответ
    if r.status_code != 200:
        print("Ответ API:", r.text)
        return

    download_url = r.json().get("href")

    if not download_url:
        print("❌ Нет ссылки на скачивание")
        print(r.json())
        return

    file_r = requests.get(download_url, stream=True)

    path = os.path.join(SAVE_PATH, FILENAME)

    with open(path, "wb") as f:
        for chunk in file_r.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"✅ Скачано в {path}")


if __name__ == "__main__":
    download()