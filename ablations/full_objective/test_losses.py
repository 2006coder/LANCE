import unittest

import torch
from common import PACKAGE, read, study
from LANCE.dmpnn import mol2graph
from LANCE.losses import (
    LANCELossWeights,
    jepa_contrastive_loss,
    jepa_prediction_loss,
    lance_total_loss,
    sigreg_gaussian_loss,
)
from LANCE.model import LANCEConfig, LANCEModel
from LANCE.lance7r.incumbent import (
    LANCEConfig as FrozenConfig,
    LANCEModel as FrozenModel,
)

torch.set_num_threads(1)


def batch():
    return dict(
        protein=torch.randn(6, 5, 12),
        protein_mask=torch.ones(6, 5, dtype=torch.bool),
        bmg=mol2graph(["CC", "CO", "CN"] * 2),
        product_bmg=mol2graph(["CO", "CN", "CC"] * 2),
        task_idx=torch.tensor([0, 1, 2] * 2),
        condition=torch.randn(6, 4),
        target_z=torch.linspace(-1, 1, 6),
        target_mask=torch.ones(6, dtype=torch.bool),
        jepa_mask=torch.tensor([1, 0, 0] * 2, dtype=torch.bool),
        cluster_id=torch.arange(6),
        substrate_desc=torch.zeros(6, 8),
        ec_level1=torch.tensor([1, 2, 3] * 2),
        ec_mask=torch.ones(6, dtype=torch.bool),
        reaction_group_id=torch.tensor([1, 2, 3] * 2),
        target_scale=torch.ones(6),
    )


def loss(out, b, config):
    return lance_total_loss(
        out,
        b["target_z"],
        target_mask=b["target_mask"],
        weights=LANCELossWeights(**config["weights"]),
        jepa_mask=b["jepa_mask"],
        task_idx=b["task_idx"],
        cluster_id=b["cluster_id"],
        substrate_desc_target=b["substrate_desc"],
        ec_target=b["ec_level1"],
        ec_mask=b["ec_mask"],
        reaction_group_id=b["reaction_group_id"],
        target_scale=b["target_scale"],
    )


class LossTests(unittest.TestCase):
    def test_definitions_and_frozen_model_parity(self):
        configs = {
            n: read(PACKAGE / "configs" / f"{n}.json") for n in study()["variants"]
        }
        original = read(PACKAGE / "assets/reference/configs/seed0.json")[
            "model_config"
        ]["incumbent_config"]
        for config in configs.values():
            self.assertEqual(config["model"], original)
        small = dict(
            original,
            protein_dim=12,
            model_dim=16,
            action_dim=8,
            hidden_dim=24,
            dmpnn_hidden=16,
            task_action_adapter_dim=16,
            dropout=0.0,
        )
        torch.manual_seed(11)
        model = LANCEModel(LANCEConfig(**small))
        frozen = FrozenModel(FrozenConfig(**small))
        frozen.load_state_dict(model.state_dict(), strict=True)
        b = batch()
        model.eval()
        frozen.eval()
        self.assertTrue(torch.equal(model(b)["mu"], frozen(b)["mu"]))
        without_products = dict(b, product_bmg=None)
        self.assertTrue(torch.equal(model(b)["mu"], model(without_products)["mu"]))
        out = model(b)
        binding_mask = b["target_mask"] & (b["task_idx"] != 0)
        binding = (
            0.15
            * jepa_prediction_loss(
                out["binding_state_pred"], out["binding_state_target"], binding_mask
            )
            + 0.10
            * jepa_prediction_loss(
                out["binding_delta"], out["binding_delta_target"], binding_mask
            )
            + 0.02
            * jepa_contrastive_loss(
                out["binding_state_pred"], out["binding_state_target"], binding_mask
            )
        )
        reaction = (
            0.35
            * jepa_prediction_loss(
                out["reaction_state_pred"], out["reaction_state_target"], b["jepa_mask"]
            )
            + 0.25
            * jepa_prediction_loss(
                out["reaction_delta"], out["reaction_delta_target"], b["jepa_mask"]
            )
            + 0.05
            * jepa_contrastive_loss(
                out["reaction_state_pred"], out["reaction_state_target"], b["jepa_mask"]
            )
        )
        full_weights = read(PACKAGE / "assets/full_loss_weights.json")
        torch.manual_seed(45)
        full = loss(out, b, {"weights": full_weights})
        torch.manual_seed(45)
        regularizer = sigreg_gaussian_loss(
            [
                out["sigreg_molecule_state"],
                out["sigreg_binding_state"],
                out["sigreg_catalytic_state"],
                out["sigreg_reaction_delta"][b["jepa_mask"]],
                out["sigreg_reaction_state_pred"],
                out["sigreg_product_state"][b["jepa_mask"]],
                out["sigreg_reaction_state_target"][b["jepa_mask"]],
                out["sigreg_binding_state_pred"][binding_mask],
                out["sigreg_binding_state_target"][binding_mask],
                out["sigreg_binding_delta"][binding_mask],
                out["sigreg_binding_delta_target"][binding_mask],
            ]
        )
        self.assertTrue(torch.equal(full["sigreg"], regularizer.detach()))
        expected = {
            "full_no_jepa": full["loss"] - binding - reaction,
            "full_no_sigreg": full["loss"] - 0.02 * regularizer,
            "full_no_jepa_no_sigreg": (
                full["loss"] - binding - reaction - 0.02 * regularizer
            ),
        }
        self.assertEqual(
            set(study()["removed_losses"]["full_no_jepa"]),
            {
                "jepa",
                "jepa_delta",
                "jepa_contrastive",
                "binding_jepa",
                "binding_jepa_delta",
                "binding_jepa_contrastive",
            },
        )
        self.assertEqual(study()["removed_losses"]["full_no_sigreg"], ["sigreg"])
        self.assertEqual(
            set(study()["removed_losses"]["full_no_jepa_no_sigreg"]),
            set(study()["removed_losses"]["full_no_jepa"]) | {"sigreg"},
        )
        parameters = list(model.parameters())
        for name, value in expected.items():
            removed = set(study()["removed_losses"][name])
            for term, original_weight in full_weights.items():
                expected_weight = 0.0 if term in removed else original_weight
                self.assertEqual(configs[name]["weights"][term], expected_weight, term)
            torch.manual_seed(45)
            actual = loss(out, b, configs[name])["loss"]
            self.assertTrue(torch.allclose(actual, value, atol=1e-6), name)
            actual_grad = torch.autograd.grad(
                actual, parameters, retain_graph=True, allow_unused=True
            )
            expected_grad = torch.autograd.grad(
                value, parameters, retain_graph=True, allow_unused=True
            )
            for param, left, right in zip(parameters, actual_grad, expected_grad):
                left = torch.zeros_like(param) if left is None else left
                right = torch.zeros_like(param) if right is None else right
                self.assertTrue(torch.allclose(left, right, atol=2e-6, rtol=1e-4), name)
        self.assertGreater(configs["full_no_jepa"]["weights"]["sigreg"], 0)
        self.assertGreater(configs["full_no_sigreg"]["weights"]["binding_jepa"], 0)
        self.assertGreater(configs["full_no_sigreg"]["weights"]["jepa"], 0)


if __name__ == "__main__":
    unittest.main()
