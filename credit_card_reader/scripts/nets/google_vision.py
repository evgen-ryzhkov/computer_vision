"""
Optical character recognition (OCR) by Google vision API

"""

import config.access as access_config
import config.main as main_config
import requests


class GoogleVision:

    def read_text_from_image(self, image_base_64):
        request_params = {'key': access_config.GOOGLE_VISION_API_KEY}
        body = self._make_request(image_base_64)
        response = requests.post(url=main_config.VISION_API_URL, params=request_params, json=body)
        response_json = response.json()

        try:
            text = response_json['responses'][0]['textAnnotations'][0]['description']
        except:
            text = ''
        return text

    @staticmethod
    def _make_request(image_base_64):
        # image_base_64 = self._convert_img_to_base64(image)
        # languageHints needs because sometimes google vision tryied read numbers as not English characters (Cyrillic for instance)
        return {
          "requests": [
            {
              "features": [
                {
                   "maxResults": 50,
                   "type": "DOCUMENT_TEXT_DETECTION"
                }
              ],
              "image": {
                 'content': image_base_64
              },
              "imageContext": {
                  "languageHints": ["en"],
              }
            }
          ]
        }
