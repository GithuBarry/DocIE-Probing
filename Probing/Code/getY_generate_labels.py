import json
import nltk
import numpy as np

if __name__ == '__main__':

    muc_1700_input = open("../../Corpora/MUC/muc/processed2/muc_1700_GTT_style-test-dev-train.json")
    examples = json.load(muc_1700_input)
    muc_1700_input.close()

    num_templates = [len(example['templates']) for example in examples]
    np.save("Y_num_events_muc1700", np.array(num_templates))

    num_words = [len(nltk.word_tokenize(example['doctext'])) for example in examples]
    np.save("Y_num_tokens_muc1700", np.array(num_words))

    num_sent = [len(nltk.sent_tokenize(example['doctext'])) for example in examples]
    np.save("Y_num_sent_muc1700", np.array(num_sent))