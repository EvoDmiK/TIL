from sklearn.metrics import pairwise_distances
import numpy as np

from misc.sampling_def import SamplingMethod


class KCenterGreedy(SamplingMethod):
    
    def __init__(self, x, y, seed, metric = 'euclidean'):
        
        self.x                = x
        self.y                = y
        self.n_obs            = self.x.shape[0]
        self.metric           = metric
        self.flat_x           = self.flatten_x()
        self.features         = self.flat_x
        self.min_dist         = None
        self.already_selected = []
        
        
    def update_distances(self, cluster_centers, only_new = True, reset_dist = False):
        
        if reset_dist: self.min_dist = None
        if   only_new: cluster_centers = [d for d in cluster_centers
                                          if d not in self.already_selected]
        
        if cluster_centers:
            
            x    = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric = self.metric)
            
            self.min_dist = np.min(dist, axis = 1).reshape(-1, 1) if self.min_dist is None \
                            else np.minimum(self.min_dist, dist)     
            
            
    def select_batch_(self, model, already_selected, N, **kwargs):
        
        try:
            print('Getting transformed features...')
            self.features = model.transform(self.x)
            print('Compute distances...')
            
            self.update_distances(already_selected, only_new = False, reset_dist = True)
            
        except:
            
            print('Using flat_X as features')
            self.update_distances(already_selected, only_new = True, reset_dist = False)
            
            
        new_batch = []
        
        for _ in range(N):
            
            idx = np.random.choice(np.arange(self.n_obs)) if self.already_selected is None \
                  else np.argmax(self.min_distances)
            
            assert idx not in already_selected
            self.update_distances([idx], only_new = True, reset_dist = False)
            new_batch.append(idx)
            
        print(f'Maximum distance from cluster centers is {self.min_dist:.2f}')
        self.already_selected = already_selected
        
        
        return new_batch
