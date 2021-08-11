To run cyclegan type:
python cyclegan.py <your_model_name> <pretrained_model_name>

For example, you can train the model on cezanne dataset by changing "monet_folder" to cezanne, and typing:
python cyclegan.py cezanne_model

Later, you can change back the "monet_folder" to monet30, and use the pretrained framework by typing:
python cyclegan.py monet_transfer_from_cezanne cezanne_model



To run StarGan type:
python stargan.py <your_model_name>