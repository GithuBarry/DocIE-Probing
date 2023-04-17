import json

import nltk
import numpy as np
import pandas as pd

if __name__ == '__main__':
    muc_1700_input = open("../../Corpora/MUC/muc/processed2/muc_1700_GTT_style-test-dev-train.json")
    examples = json.load(muc_1700_input)
    muc_1700_input.close()

    num_templates = [len(example['templates']) for example in examples]
    np.save("Y_num_events_muc1700", np.array(num_templates))

    cuts = pd.qcut(num_templates, q=10, duplicates='drop').categories
    num_templates_10 = [[nt in cut for cut in cuts].index(True) for nt in num_templates]
    np.save("Y_bucket_num_events_bucket_muc1700", np.array(num_templates))

    num_words = [len(nltk.word_tokenize(example['doctext'])) for example in examples]
    np.save("Y_num_tokens_muc1700", np.array(num_words))

    cuts = pd.qcut(num_words, q=10, duplicates='drop').categories
    num_words_10 = [[nt in cut for cut in cuts].index(True) for nt in num_words]
    np.save("Y_bucket_num_tokens_muc1700", np.array(num_words_10))

    num_sent = [len(nltk.sent_tokenize(example['doctext'])) for example in examples]
    np.save("Y_num_sent_muc1700", np.array(num_sent))

    cuts = pd.qcut(num_sent, q=10, duplicates='drop').categories
    num_sent_10 = [[nt in cut for cut in cuts].index(True) for nt in num_sent]
    np.save("Y_bucket_num_sent_muc1700", np.array(num_sent_10))
