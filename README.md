# Ethicsbot

Ethicsbot is a robot that can answer ethical questions "AmITheAsshole" on Reddit . 

It is made up of models for two purposes: first, a multi-class classifier (via [distilled BERT](https://arxiv.org/abs/1910.01108)) to label submissions with "YTA" or "NTA" or "NAH", 
and second, a seq2seq model (via [BART](https://arxiv.org/abs/1910.13461)) to generate an explanatory text for the predicted label. 

Our multi-class classifier accuracy is pretty good, and the seq2seq model produces generally coherent and logical explanations in response to submissions.
