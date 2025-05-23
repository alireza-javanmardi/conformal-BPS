from epiuc.uncertainty.wrapper.evidential import Evidential_Classifier
from epiuc.uncertainty.wrapper.mc_dropout import MC_Classifier
from epiuc.uncertainty.wrapper.ensemble import Ensemble_Classifier
from epiuc.uncertainty.classification import Resnet50_Cifar100, Resnet18_Cifar100
from epiuc.utils.data_load import load_cifar100
from epiuc.utils.general import set_seeds

if __name__ == "__main__":
    set_seeds(42)
    trainloader, valloader, testloader = load_cifar100(vali_size=0.3)
    resnet = Resnet18_Cifar100(random_state=42, pretrained=True)
    # print(resnet)
    evidential_resnet = Evidential_Classifier(
        base_model=resnet,
        evidence_method="softplus",
        kl_reg_scaler=0.001,
        random_state=42,
    )
    mc_resnet = MC_Classifier(
        base_model=resnet, n_iterations=10, dropout_prob=0.05, random_state=42
    )
    ensemble_resnet = Ensemble_Classifier(
        base_model=resnet, n_models=5, random_state=42
    )
    print(evidential_resnet)
    evidential_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)
    mc_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)
    # ensemble_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)

    resnet = Resnet50_Cifar100(random_state=42, pretrained=True)
    # print(resnet)
    evidential_resnet = Evidential_Classifier(
        base_model=resnet,
        evidence_method="softplus",
        kl_reg_scaler=0.001,
        random_state=42,
    )
    mc_resnet = MC_Classifier(
        base_model=resnet, n_iterations=10, dropout_prob=0.05, random_state=42
    )
    ensemble_resnet = Ensemble_Classifier(
        base_model=resnet, n_models=5, random_state=42
    )
    evidential_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)
    mc_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)
    # ensemble_resnet.fit(trainloader, dataset_name="CIFAR100", n_epochs=50)
