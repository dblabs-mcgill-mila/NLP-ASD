import torch

def create_batches(dataset, batch_size):
    """
    Create batches from a dataset (not random)

    Args:
        dataset (torch.Tensor): The dataset to split into batches
        batch_size (int): The size of each batch
    Returns:
        batches (list): A list of batches
    """
    # Calculate the total number of data points in the dataset
    num_data_points = dataset.shape[0]
    # Calculate the number of batches needed
    num_batches = (num_data_points + batch_size - 1) // batch_size
    # Split the data into batches using torch.split
    batches = torch.split(dataset, batch_size)
    
    return batches