import torch

from extractive_finetuning import BertForQuestionAnswering
from main_hybrid_decoder import DecoderConfig as HybridDecoderConfig
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import BertEncoder, BertForMLM, ModelConfig
from standard_generative_decoder import DecoderConfig, GenerativeQAModel

VOCAB_SIZE = 100
HIDDEN = 32
SEQ = 16
BATCH = 2


def tiny_encoder_cfg():
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )


def tiny_decoder_cfg():
    return DecoderConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=SEQ,
    )


def encoder_inputs():
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))
    token_type_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    return input_ids, token_type_ids, attention_mask


def test_encoder_output_shape():
    model = BertEncoder(tiny_encoder_cfg())
    input_ids, token_type_ids, attention_mask = encoder_inputs()
    out = model(input_ids, token_type_ids, attention_mask)
    assert out.shape == (BATCH, SEQ, HIDDEN)


def test_mlm_loss_is_finite():
    model = BertForMLM(tiny_encoder_cfg())
    input_ids, token_type_ids, attention_mask = encoder_inputs()
    labels = input_ids.clone()
    labels[:, :4] = -100
    logits, loss = model(input_ids, token_type_ids, attention_mask, labels=labels)
    assert logits.shape == (BATCH, SEQ, VOCAB_SIZE)
    assert torch.isfinite(loss)


def test_extractive_qa_logits_shape():
    model = BertForQuestionAnswering(tiny_encoder_cfg())
    input_ids, token_type_ids, attention_mask = encoder_inputs()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    assert out["start_logits"].shape == (BATCH, SEQ)
    assert out["end_logits"].shape == (BATCH, SEQ)
    assert "loss" not in out


def test_extractive_qa_loss_is_finite():
    model = BertForQuestionAnswering(tiny_encoder_cfg())
    input_ids, token_type_ids, attention_mask = encoder_inputs()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        start_positions=torch.ones(BATCH, dtype=torch.long),
        end_positions=torch.full((BATCH,), 2, dtype=torch.long),
    )
    assert torch.isfinite(out["loss"])


def test_hybrid_decoder_logits_shape_and_loss():
    model = GenerativeQAModelHybrid(
        tiny_encoder_cfg(),
        HybridDecoderConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN,
            num_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=SEQ,
        ),
    )
    enc_ids, enc_ttype, enc_mask = encoder_inputs()
    dec_len = 8
    dec_ids = torch.randint(0, VOCAB_SIZE, (BATCH, dec_len))
    dec_mask = torch.ones(BATCH, dec_len, dtype=torch.long)
    labels = dec_ids.clone()

    out = model(
        encoder_input_ids=enc_ids,
        encoder_token_type_ids=enc_ttype,
        encoder_attention_mask=enc_mask,
        decoder_input_ids=dec_ids,
        decoder_attention_mask=dec_mask,
        labels=labels,
    )
    assert out["logits"].shape == (BATCH, dec_len, VOCAB_SIZE)
    assert torch.isfinite(out["loss"])


def test_standard_decoder_logits_shape_and_loss():
    model = GenerativeQAModel(tiny_encoder_cfg(), tiny_decoder_cfg())
    enc_ids, enc_ttype, enc_mask = encoder_inputs()
    dec_len = 8
    dec_ids = torch.randint(0, VOCAB_SIZE, (BATCH, dec_len))
    dec_mask = torch.ones(BATCH, dec_len, dtype=torch.long)
    labels = dec_ids.clone()

    out = model(
        encoder_input_ids=enc_ids,
        encoder_token_type_ids=enc_ttype,
        encoder_attention_mask=enc_mask,
        decoder_input_ids=dec_ids,
        decoder_attention_mask=dec_mask,
        labels=labels,
    )
    assert out["logits"].shape == (BATCH, dec_len, VOCAB_SIZE)
    assert torch.isfinite(out["loss"])


def test_generate_returns_padded_sequence():
    torch.manual_seed(0)
    model = GenerativeQAModelHybrid(
        tiny_encoder_cfg(),
        HybridDecoderConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN,
            num_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=SEQ,
        ),
    )
    model.eval()
    enc_ids, enc_ttype, enc_mask = encoder_inputs()
    out = model.generate(
        encoder_input_ids=enc_ids[:1],
        encoder_token_type_ids=enc_ttype[:1],
        encoder_attention_mask=enc_mask[:1],
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        max_new_tokens=5,
        beam_size=2,
    )
    assert out.shape[0] == 1
    assert out.shape[1] <= 6
    assert out[0, 0].item() == 2
