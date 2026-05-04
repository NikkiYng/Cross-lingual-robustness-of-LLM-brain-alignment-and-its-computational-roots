import torch
import pandas as pd
import numpy as np
import spacy_alignments as tokenizations
from tqdm import tqdm 
from transformers import BertTokenizer, BertModel


#1.  Load the models
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")
model = BertModel.from_pretrained("hfl/chinese-macbert-base", 
                                  output_hidden_states=True, 
                                  output_attentions=True)


#2.  Embedding functions 
def embedding_alignment(my_dict, my_embedding):  
    align_embeds = []
    for i in range(len(my_dict)):
        align_embed = my_embedding[my_dict[i], :].mean(axis=0)
        align_embeds.append(align_embed)  
    return np.array(align_embeds)

def mean_pooling(model_output, attention_mask): 
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def retrieve_bert_embedding(tokens):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    embeddings = {} 
    sent = ' '.join(tokens) 
    encoded_input = tokenizer(sent, add_special_tokens=True, return_tensors='pt')
    input_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])[1:-1] # remove [CLS] and [SEP]
    # alignment
    word2token, token2word = tokenizations.get_alignments(tokens, input_tokens) 
    # layer embeddings 
    output = model(input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    return_dict=False) 
    layer_embeddings = output[2][1:] # (n_layers, n_words, 768)
    # align layer embeddings to the natural tokens
    layer_embeds = [embedding_alignment(word2token, 
                                        layer_embedding.squeeze(dim=0)[1:-1].detach().numpy()) 
                    for layer_embedding in layer_embeddings] 
    # save in the dictionary 
    for i in range(len(layer_embeds)):
        embeddings[f'BERT_embed_{i+1}_isol'] = layer_embeds[i] # (n_words, 768)
    return embeddings


def get_all_embeddings(utt):
    all_embeddings = []
    for tokens in tqdm(utt):
        # Retrieve embeddings for the sentence
        embeddings = retrieve_bert_embedding(tokens)
        # Get number of words in the sentence
        num_words = len(tokens)
        # Flatten each word’s embeddings in the sentence
        for i in range(num_words):
            # Create a dictionary for each word with its embeddings
            word_embedding = {key: np.array(value[i]) for key, value in embeddings.items()}
            all_embeddings.append(word_embedding)
    
    # Convert the flattened list of word embeddings to a DataFrame
    embeddings_df = pd.DataFrame(all_embeddings)
    return embeddings_df



#3. Process by groups

df = pd.read_csv("/gpfs/home/nyang/tr/lppCNtr.csv", encoding='utf-8-sig')
word = df['word']
# Group by 'snt_id' and collect 'word' lists for each sentence in these runs
utt = df.groupby('snt_id')['word'].apply(list).tolist()
embeddings_df = get_all_embeddings(utt)

    
# Concatenate the embeddings DataFrame and the original  data along the columns axis
result = pd.concat([df, embeddings_df], axis=1)
result.columns

npz_data = {
    'word': result['word'].values,
    'offset': result['offset'].values,
    'run_id': result['run_id'].values,
}

# Convert each embedding or attention layer column to arrays to ensure consistency
for i in range(1, 13):
    npz_data[f'BERT_layer_{i}isol'] = np.array([np.array(x) for x in result[f'BERT_embed_{i}_isol']])

# Save as .npz file
np.savez("/gpfs/home/nyang/embs/bert_embs_cn.npz", **npz_data)


