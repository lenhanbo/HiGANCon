def get_model(name):
    # Keep the legacy upstream entry point without importing its optional FID stack
    # whenever a concrete network module is imported by the new trainer.
    if name == 'adversarial_model':
        from networks.model import AdversarialModel
        return AdversarialModel
    raise KeyError(name)
