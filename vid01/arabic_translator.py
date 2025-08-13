# src/nodes.py
import torch 
from server import PromptServer
from transformers import MarianMTModel, MarianTokenizer

class ArabicTranslator:
    CATEGORY = "Text"
    @classmethod    
    def INPUT_TYPES(s):
        return { "required":  { 
            "arabic_text": ("STRING", {"multiline": True}), 
            "language": (["arabic"],),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "do_translate"
    
    def __init__(self):
        super().__init__()
        print("------------------- initializing translate")
        PromptServer.instance.send_sync("text.arabictranslator.textmessage", {"message":f"------------------- initializing translate"})
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def do_translate(self, arabic_text, language):
        inputs = self.tokenizer(arabic_text, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return (translated_text,)
    
        # translation = "unknown"
        # if language == "arabic":
        #     translation = "testing translate to arabic"
        # else:
        #     raise Exception(f"translate from {language} is not implemented!")
        # translation = translation + "\n" + arabic_text
        # print("------------------- running translate")
        # PromptServer.instance.send_sync("text.arabictranslator.textmessage", {"message":f"------------------- {arabic_text} -> {translation}"})
        # return (translation,)

NODE_CLASS_MAPPINGS = {
    "ArabicTranslator" : ArabicTranslator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArabicTranslator": "ترجمة للعربي",
}