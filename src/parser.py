import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Run VGAE for Link Prediction")

    parser.add_argument("--embedding_size",
                        dest = "embedding_size",
                        type = int,
                        default = 16,
	                help = "Number of dimensions. Default is 128.")

    parser.add_argument("--epochs",
                        dest = "epochs",
                        type = int,
                        default = 300,
	                help = "Number of gradient descent iterations. Default is 300.")

    parser.add_argument("--learning_rate",
                        dest = "learning_rate",
                        type = float,
                        default = 0.01,
	                help = "Gradient descent learning rate. Default is 0.01.")

    parser.add_argument("--neurons",
                        dest = "num_neurons",
                        type = int,
                        default = 32,
	                help = "Number of neurons by hidden layer. Default is 32.")
                
    parser.add_argument("--dataset",
                        dest ="dataset",
                        default = 'karate_club',
	                help = "Name of the dataset. Default is karate_club.")

    parser.add_argument("--directory",
                        dest ="directory",
                        default = 'benchmarks',
	                help = "Name of data's container. Default is data.")

    parser.add_argument("--test_size",
                        dest = "test_size",
                        type = float,
                        default = 0.10,
	                help = "Size of test dataset. Default is 10%.")
    
    return parser.parse_args()
