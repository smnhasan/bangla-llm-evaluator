from typing import List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TranslationModel:
    """Manages loading and configuration of translation models and tokenizers."""

    def __init__(self, model_name: str, use_fast: bool = False):
        """
        Initialize the translation model and tokenizer.

        Args:
            model_name (str): Hugging Face model name (e.g., 'csebuetnlp/banglat5_nmt_bn_en').
            use_fast (bool): Whether to use fast tokenizer. Defaults to False.

        Raises:
            RuntimeError: If model or tokenizer loading fails.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer for {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def translate_batch(
            self, sentences: List[str], max_tokens: int = 128
    ) -> List[str]:
        """
        Translate a batch of sentences using the loaded model and tokenizer.

        Args:
            sentences (List[str]): List of sentences to translate.
            max_tokens (int): Maximum number of tokens to generate per sentence.

        Returns:
            List[str]: List of translated sentences.

        Raises:
            ValueError: If the input list is empty or contains invalid sentences.
            RuntimeError: If translation fails due to model inference issues.
        """
        if not sentences:
            logger.error("Input sentence list is empty")
            raise ValueError("Input sentence list cannot be empty")

        if not all(isinstance(s, str) and s.strip() for s in sentences):
            logger.error("Invalid sentences in input list")
            raise ValueError("All sentences must be non-empty strings")

        # Normalize sentences
        normalized_sentences = [normalize(sentence) for sentence in sentences]
        logger.debug(f"Normalized {len(normalized_sentences)} sentences")

        # Tokenize and pad as batch
        try:
            inputs = self.tokenizer(
                normalized_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise RuntimeError(f"Tokenization failed: {str(e)}")

        # Generate translations
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_tokens
                )
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")

        # Decode translations
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(f"Successfully translated {len(translations)} sentences")
        return translations


def load_translation_models() -> Tuple[TranslationModel, TranslationModel]:
    """
    Load Bangla-to-English and English-to-Bangla translation models.

    Returns:
        Tuple[TranslationModel, TranslationModel]: Bangla-to-English and English-to-Bangla models.

    Raises:
        RuntimeError: If any model fails to load.
    """
    try:
        bn_to_en = TranslationModel("csebuetnlp/banglat5_nmt_bn_en", use_fast=False)
        en_to_bn = TranslationModel("csebuetnlp/banglat5_nmt_en_bn", use_fast=False)
        return bn_to_en, en_to_bn
    except Exception as e:
        logger.error(f"Failed to load translation models: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


def translate_bn_to_en_batch(sentences: List[str], max_tokens: int = 128) -> List[str]:
    """
    Translates a list of Bangla sentences to English using the BanglaT5 model.

    Args:
        sentences (List[str]): List of Bangla sentences.
        max_tokens (int): Maximum number of tokens to generate per sentence.

    Returns:
        List[str]: Translated English sentences.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If translation fails.
    """
    bn_to_en_model, _ = load_translation_models()
    return bn_to_en_model.translate_batch(sentences, max_tokens)


def translate_en_to_bn_batch(sentences: List[str], max_tokens: int = 128) -> List[str]:
    """
    Translates a list of English sentences to Bangla using the BanglaT5 model.

    Args:
        sentences (List[str]): List of English sentences.
        max_tokens (int): Maximum number of tokens to generate per sentence.

    Returns:
        List[str]: Translated Bangla sentences.

    Raises:
        ValueError: If input is invalid.
        RuntimeError: If translation fails.
    """
    _, en_to_bn_model = load_translation_models()
    return en_to_bn_model.translate_batch(sentences, max_tokens)


if __name__ == "__main__":
    # Example usage
    try:
        # Test Bangla to English translation
        bangla_sentences = ["আমি বই পড়তে ভালোবাসি", "আজ আকাশ খুব সুন্দর"]
        english_translations = translate_bn_to_en_batch(bangla_sentences)
        for bn, en in zip(bangla_sentences, english_translations):
            print(f"Bangla: {bn} -> English: {en}")

        # Test English to Bangla translation
        english_sentences = ["I love reading books", "The sky is beautiful today"]
        bangla_translations = translate_en_to_bn_batch(english_sentences)
        for en, bn in zip(english_sentences, bangla_translations):
            print(f"English: {en} -> Bangla: {bn}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error during translation: {str(e)}")