import joblib
import os
import numpy
import sklearn.mixture
import torch
from tqdm import tqdm

class LGMM(object):
    """ 
    Gaussian mixture model on the latent space. 
    It predicts the out-of-distribution score given 
    the latent representation of the scene encoder of
    the trajectory prediction model.  

    """

    def __init__(
        self,
        model_path: str = None,
        K=6,
    ) -> None:

        self.model_path = None
        self.model = None
        self.K = K
        self.covar_type = "full"
        self.model_root = os.path.join(
            os.getcwd(), "experiments", "ood_detection", "001_lgmm")
        self.verbose = 2
        self.verbose_interval = 10

        if model_path:
            self.set_model_path(model_path)
            self.load()
        else:
            self.initialize()

    def set_model_path(self, model_path):
        self.model_path = model_path

    def create_model_path(self):
        model_dir = "train_lgmm_K{}".format(
            self.K)
        model_file = "lgmm.joblib"
        self.model_path = os.path.join(self.model_root, model_dir, model_file)

    def load(self):
        self.model = joblib.load(self.model_path)

    def initialize(self):
        """Initialize sklearn GMM.
        """
        self.model = sklearn.mixture.GaussianMixture(
            n_components=self.K,
            covariance_type=self.covar_type,
            random_state=0,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval
        )

    def train(self, dataloader, model, dataloader_idx=0):
        latent_features = []
        model.eval()
        print("Collecting Features...")
        # use tqdm to show progress bar
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=dataloader_idx)
                enc_dict = model.context_encoder(batch)
                features = enc_dict['center_objects_feature'].cpu().numpy()
            latent_features.append(features)
        latent_features = numpy.concatenate(latent_features, axis=0)
        latent_features = latent_features[:10000]
        # latent_features = latent_features[:1000]
        print("Fitting GMM...")
        self.model.fit(latent_features)

    def export_model(self, export_path=None):
        if export_path:
            joblib.dump(self.model, export_path)
        else:
            self.create_model_path()
            os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def predict_ood_score(self, features: numpy.ndarray) -> numpy.ndarray:
        """Method to compute the negative log-likelihood of each feature vector.

        Args:
            features: Latent features [NxF] with number of samples N and 
                feature dimension F.
        Return:
            Out-of-distribution score [N]
        """
        alpha_hat = -self.model.score_samples(features)
        return alpha_hat
    
    def get_distance(self, x):
        """Compute the minimum Mahalanobis distance from x to each GMM component."""
        distances = []
        for mean, covar in zip(self.model.means_, self.model.covariances_):
            diff = x - mean
            inv_covar = numpy.linalg.inv(covar)
            distance = numpy.sqrt(numpy.dot(numpy.dot(diff, inv_covar), diff.T))
            distances.append(distance)
        return numpy.min(distances, axis=0)