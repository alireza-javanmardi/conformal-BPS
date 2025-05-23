from epiuc.uncertainty.wrapper.evidential import Evidential_Classifier
from epiuc.uncertainty.wrapper.ensemble import Ensemble_Classifier
from epiuc.uncertainty.wrapper.mc_dropout import MC_Classifier
from epiuc.uncertainty.classification import Resnet50_Cifar10, Resnet18_Cifar10
from epiuc.utils.data_load import load_cifar10
from epiuc.utils.general import set_seeds

if __name__ == "__main__":
    set_seeds(42)
    trainloader, _, testloader = load_cifar10(vali_size=0)

    resnet = Resnet18_Cifar10(random_state=42, pretrained=True)
    evidential_resnet = Evidential_Classifier(
        base_model=resnet, evidence_method="softplus", kl_reg_scaler=1, random_state=42
    )
    mc_resnet = MC_Classifier(
        base_model=resnet, n_iterations=10, dropout_prob=0.05, random_state=42
    )
    ensemble_resnet = Ensemble_Classifier(
        base_model=resnet, n_models=5, random_state=42
    )

    evidential_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)
    # ensemble_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)
    mc_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)

    resnet = Resnet50_Cifar10(random_state=42, pretrained=True)
    evidential_resnet = Evidential_Classifier(
        base_model=resnet, evidence_method="softplus", kl_reg_scaler=1, random_state=42
    )
    mc_resnet = MC_Classifier(
        base_model=resnet, n_iterations=10, dropout_prob=0.05, random_state=42
    )
    ensemble_resnet = Ensemble_Classifier(
        base_model=resnet, n_models=5, random_state=42
    )
    evidential_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)
    # ensemble_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)
    mc_resnet.fit(trainloader, dataset_name="CIFAR10", n_epochs=50)
