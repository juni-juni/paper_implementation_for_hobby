# Learning Conditioned Graph Structures forInterpretable Visual Question Answering

## for word embedding 
- pretrained a dynamic Gated Recurrent Unit (hidden size of 1024)

## model

### Graph learner
* aims to learn adjacency matrix
* concat question embedding to N visual features
* nth_e = non_linear_transform(torch.cat(word_embedding, nth_visual_feature))
* non_linear_transform = two dense linear layer of size 512
* A = matmul(E, E_transposed), where E is matrix of e
* N(i) = topm(a_i)
* m=16

### Spatial graph convolution (L times, L=2)
* graph convolution to learn new object representation
* f_k(i) = sum_of_neighbor(gaussian_kerel(polar_coordinate)*v_j*softmax(a_ij))
* h_i = concat(f_1(i),...,f_k(i))
* d_h = [2048, 1024]
* activation functin of all dense and convolution layers is ReLU
* k=8

### prediction layers
* max_pooling layers across the node dimension
* output = 2_layers_mlp(question_embedding * h_max) # with relu activation

## train
* loss
    - Cross Entropy Loss
    - target = number_of_votes  [n # what is number of votes??]
* optimizer Adam with learning rate 0.0001 -> 0.00005 after 30 epoch
* batch_size = 64
* 35 epoch
* use dropout for image features and all dense layers exept final one
## Dataset
* VQA 2.0 dataset
* [train : 40%, val : 20%, test : 40%]
* 36 object bounding boxes with (2048) + (bouding box corner) =>2052 dimension
* normalize bounding box with interval [0, 1] 