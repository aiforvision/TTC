from typing import List
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity


def random_points_on_n_sphere(n, num_points=1):
    """
    Generate uniformly distributed random points on the unit (n-1)-sphere.
    (Marsaglia, 1972) algorithm

    :param n: Dimension of the space
    :param num_points: Number of points to generate
    :return: Array of points on the (n-1)-sphere
    """
   
    
    x = np.random.normal(0, 1, (num_points, n))

    # Calculate the radius for each point
    radius = np.linalg.norm(x, axis=1).reshape(-1, 1)

    
    # Normalize each point to lie on the unit n-sphere
    points_on_sphere = x / radius

    return torch.from_numpy(points_on_sphere)





def assign_to_prototypes(projections, prototypes):
    """ Assign each projection to the closest prototype using pairwise cosine similarity. """
    prototype_tensor = torch.stack([p.data for p in prototypes])
    cosine_sim = pairwise_cosine_similarity(projections, prototype_tensor)
    # Assign each projection to the prototype with the highest similarity
    assignments = torch.argmax(cosine_sim, dim=1)
    return assignments

def find_n_prototypes(n, projections):
    print(f"searching {n} prototypes")
    dim = projections.shape[1]
    #
    # random_vectors_on_sphere = random_points_on_n_sphere(dim, num_points=n).float()



    # find vector with smalles similiarity:

    vector = nn.Parameter(torch.randn(projections.shape[1]))

    optimizer = torch.optim.Adam([vector], lr=0.0001)

    for e in range(10000):   # number of iterations
        optimizer.zero_grad()  
        # cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(projections, vector.unsqueeze(0).expand_as(projections), dim=1)
        
        # minimize the sum (or average) of the cosine similarity
        loss = torch.sum(cos_sim)
        loss.backward()  
        optimizer.step()
        
        # renormalize vector back to magnitude 1
        vector.data = vector.data / torch.norm(vector.data)

        if e % 500 == 0:
            print('epoch', e, 'loss', loss.item())
    if n == 1:
        return vector.data
    if n == 2:
        return torch.stack([vector.data, -vector.data])
    
    raise NotImplementedError("n>2 not implemented")
