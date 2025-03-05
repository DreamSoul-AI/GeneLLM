import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel

# dataset descriptions

EMP = """
Epigenetic marks prediction (Yeast) predicts epigenetic marks in terms of histone modification in yeast, modifications on the histone that influence the structure of genome, then subsequently regulate gene expression without altering the DNA sequence. Precise prediction of these marks aids in elucidating the role of non-coding genome segments in yeast. We download the 10 datasets and randomly split each dataset into training, validation, and test sets with a ratio of 8:1:1. 
The labels represent the presence (True) or absence (False) of a specific histone modification in the yeast genome.
"""

mouse = """
Transcription factor binding site prediction (Mouse) predicts the binding site of transcription factors on mouse genomes. Similar to human binding site data, we obtain mouse ENCODE ChIP-seq data for specific transcription factors(Stamatoyannopoulos et al., 2012), which is the largest available collection on the UCSC genome browser (n=78). This time, the negative examples are created using dinucleotide shuffling while preserving random and/or general relative frequencies, in other words no transcription factor binding, while all other settings stay the same as the human TFBS prediction dataset. We also randomly select 5 datasets out of the 78 datasets using the same process described above.
The labels represent whether a sequence is a transcription factor binding site (True) or not (False).
"""

prom300 = """
Promoter detection (Human) focuses on identifying (proximal) promoter regions, crucial sequences in the human genome responsible for binding transcription factors to initiate gene transcription. As many primary regulatory elements are located in this region, accurately detecting these sites is instrumental in advancing our grasp of gene regulation mechanisms and pinpointing the genomic underpinnings of numerous diseases. The dataset is divided two fold, TATA and non-TATA, based on whether a TATA box motif is present in the sequence. We extract -249 ~ +50 bp around the transcription start site (TSS) from TATA and non-TATA promoters downloaded from Eukaryotic Promoter Database (EPDnew) (Dreoset al., 2013) and use it as our promoter class. Meanwhile, we construct the non-promoter class with equal-sized randomly selected sequences outside of promoter regions but with TATA motif (TATA non-promoters) or randomly substituted sequences (non-TATA, non-promoters). We also combine the TATA and non-TATA datasets to obtain a combined dataset named all.
The labels represent a sequence being a promoter (True) or not being a promoter (False).
"""

promcore = """
Core promoter detection (Human) focuses on identifying (core) promoter regions in human, crucial sequences in the human genome responsible for regulating transcription. As many primary regulatory elements are located in this region, accurately detecting these sites is instrumental in advancing our grasp of gene regulation mechanisms and pinpointing the genomic underpinnings of numerous diseases. The dataset is divided twofold, TATA and non-TATA, based on whether a TATA box motif is present in the sequence. This task is focusing on predicting the core promoter region only, the central region closest to the transcription start site (TSS) and start codon. A much shorter context window (center -34 ~ +35 bp around TSS) is provided from TATA and non-TATA promoters downloaded from Eukaryotic Promoter Database (EPDnew) (Dreoset al., 2013) and use it as our promoter class, making this a more challenging task than proximal promoter prediction. Meanwhile, we construct the non-promoter class with equal-sized randomly selected sequences outside of promoter regions but with TATA motif (TATA non-promoters) or randomly substituted sequences (non-TATA, non-promoters). We also combinethe TATA and non-TATA datasets to obtain a combined dataset named all.
The labels represent a sequence being a promoter (True) or not being a promoter (False) in Human.
"""

splice = """
Splice site prediction (Human) predicts splice donor and acceptor sites, which are the exact locations in the human genome where alternative splicing occurs. This prediction is crucial to understanding protein diversity and the implications of aberrant splicing in genetic disorders. The dataset (Wang et al., 2019) consists of 400-bp-long sequences extracted from Ensembl GRCh38 human reference genome. As suggested by Ji et al. (2021), existing models can achieve almost perfect performance on the original dataset for splicing prediction, containing 10,000 splice donors, acceptors, and non-splicesite sequences, which is overly optimistic on detecting non-canonical splice sites in reality. As such, we reconstruct the dataset by iteratively adding adversarial examples (unseen false positive predictions in hold-out set) in order to make this task more challenging.
The labels represent whether a sequence is a splice donor/acceptor site (True) or not (False).
"""

tf = """
Transcription factor binding site prediction (Human) predicts binding sites of transcription factors (TF), the key proteins that regulate gene expression in the human genome. Their accurate prediction is key to deciphering complex TF-gene interactions and identifying potential transcriptional targets for specific transcription factors. We accessed the legacy 690 ENCODE ChIP-seq experiments (Consortium et al., 2012) via the UCSC genome browser, which encompasses 161 TF binding profiles in 91 human cell lines. We extracted a 101-bp region around the center of each peak as TFBS class and nonoverlapping sequences with the same length and GC content as non-TFBS class. Finally, we randomly select 5 datasets out of a subset of 690 that we curated by heuristically filtering out tasks that are either too trivial (e.g., over 0.95 F1) or too challenging (e.g., less than 0.50 F1) for existing language models.
The labels represent whether a sequence is a transcription factor binding site (True) or not (False).
"""

virus = """
Covid variant prediction (Virus) aims to predict the variant type of the SARS_CoV_2 virus based on 1000-length genome sequences. We download the genomes from the EpiCoV database (Khareet al., 2021) of the Global Initiative on Sharing Avian Influenza Data (GISAID). We consider 9 types of SARS_CoV_2 variants, including Alpha, Beta, Delta, Eta, Gamma, Iota, Kappa, Lambda and Zeta.
The labels represent the variant type of the SARS-CoV-2 virus, including Alpha, Beta, Delta, Eta, Gamma, Iota, Kappa, Lambda, and Zeta, based on 1000-length genome sequences.
"""


tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')


batch_input_texts = [EMP, mouse, prom300, promcore, splice, tf, virus]



# Tokenize the input texts
batch_dict = tokenizer(batch_input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state


# create embedding dic embeddings_named 
dataset_names = ['EMP', 'mouse', 'prom300', 'promcore', 'splice', 'tf', 'virus']
embeddings_named = {name:  # name of the dataset
                    {'embedding_seq': embeddings[i],   # embedding seq with paddings
                     'attention_mask': batch_dict['attention_mask'][i],  # attention_mask for this embedding seq 
                     'pooler_output': outputs.pooler_output[i]}  # pooler_output for this embedding seq(CLS embedding)  
                    for i, name in enumerate(dataset_names)}

# save embeddings_named
torch.save(embeddings_named, "/workspaces/GeneLLM/src/dataset/description_embedding/embeddings.pth")

