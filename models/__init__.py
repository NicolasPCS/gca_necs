from models.gca import GCA
from models.cgca_transition import CGCATransitionModel, CGCATransitionConditionModel
from models.cgca_autoencoder import CGCAAutoencoder

#print(GCA.name, GCA)

MODEL = {
    GCA.name: GCA,
	CGCATransitionModel.name: CGCATransitionModel,
	CGCATransitionConditionModel.name: CGCATransitionConditionModel,
	CGCAAutoencoder.name: CGCAAutoencoder,
}

#print(MODEL)
#print(MODEL['gca'])