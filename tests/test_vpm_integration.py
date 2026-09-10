"""
Unit tests for VPM integration and IntegratedAssessmentEngine facade.
"""

import os
import tempfile
import pytest

from pgloop import IntegratedAssessmentEngine, get_pathway, list_pathways
from pgloop.pathways import (
    CementPathway,
    ChemicalRecoveryPathway,
    ConstructionMaterialsPathway,
    REEExtractionPathway,
    SulfurAcidPathway,
    StackDisposalPathway,
    SoilAmendmentPathway,
)
from pgloop.pathways.vpms import (
    AlphaHemihydrateVPM,
    AmmonoCarbonationVPM,
    CarbothermicVPM,
    HydrationVPM,
    REEExtractionVPM,
)
from pgloop.visualization.export import ReportExporter


class TestVPMBindings:
    """Test that all valorization pathways are correctly bound to their respective VPMs."""

    def test_vpm_class_bindings(self):
        assert SulfurAcidPathway.vpm_class == CarbothermicVPM
        assert CementPathway.vpm_class == AlphaHemihydrateVPM
        assert ChemicalRecoveryPathway.vpm_class == AmmonoCarbonationVPM
        assert ConstructionMaterialsPathway.vpm_class == HydrationVPM
        assert REEExtractionPathway.vpm_class == REEExtractionVPM

    def test_static_pathways_have_no_vpm(self):
        assert StackDisposalPathway.vpm_class is None
        assert SoilAmendmentPathway.vpm_class is None

    def test_dynamic_vpm_evaluation_carbothermic(self):
        pathway = SulfurAcidPathway()
        res = pathway.evaluate_operating_conditions({"temperature_c": 1050.0, "residence_time_min": 45.0})
        assert res["status"] == "simulated"
        assert res["vpm_id"] == "VPM_carbothermic_reduction"
        assert len(res["governing_equations"]) == 3
        assert "validation" in res
        assert "is_valid" in res["validation"]
        assert "metrics" in res["validation"]

    def test_dynamic_vpm_evaluation_alpha_hemihydrate(self):
        pathway = CementPathway()
        res = pathway.evaluate_operating_conditions({"temperature_c": 125.0, "residence_time_min": 50.0})
        assert res["status"] == "simulated"
        assert res["vpm_id"] == "VPM_alpha_hemihydrate"
        assert len(res["governing_equations"]) == 4
        assert res["validation"]["is_valid"] is True

    def test_dynamic_vpm_evaluation_ammono_carbonation(self):
        pathway = ChemicalRecoveryPathway()
        res = pathway.evaluate_operating_conditions({"temperature_c": 50.0, "nh3_pg_ratio": 2.0})
        assert res["status"] == "simulated"
        assert res["vpm_id"] == "VPM_ammono_carbonation"
        assert len(res["governing_equations"]) == 4

    def test_dynamic_vpm_evaluation_hydration(self):
        pathway = ConstructionMaterialsPathway()
        res = pathway.evaluate_operating_conditions()
        assert res["status"] == "simulated"
        assert res["vpm_id"] == "VPM_hydration"

    def test_dynamic_vpm_evaluation_ree_extraction(self):
        pathway = REEExtractionPathway()
        res = pathway.evaluate_operating_conditions()
        assert res["status"] == "simulated"
        assert res["vpm_id"] == "VPM_ree_extraction"

    def test_static_fallback_evaluation(self):
        pathway = StackDisposalPathway()
        res = pathway.evaluate_operating_conditions()
        assert res["status"] == "static_fallback"
        assert res["vpm_id"] is None
        assert "parameters" in res


class TestIntegratedAssessmentEngine:
    """Test IntegratedAssessmentEngine facade."""

    def test_assess_vpm_pathway(self):
        engine = IntegratedAssessmentEngine(country="China")
        result = engine.assess("PG-CementProd")
        assert result["pathway_code"] == "PG-CementProd"
        assert result["trl"] == 9
        assert "lca" in result
        assert "tea" in result
        assert "vpm" in result
        assert "risk" in result

        # Check realistic values
        assert result["tea"]["clcc"] > 0
        assert result["tea"]["clcc"] < 50  # Cement should be well under $50/t
        assert result["vpm"]["status"] == "simulated"
        assert result["risk"]["overall_score"] >= 0

    def test_assess_static_pathway(self):
        engine = IntegratedAssessmentEngine(country="China")
        result = engine.assess("PG-Stack")
        assert result["pathway_code"] == "PG-Stack"
        assert result["vpm"]["status"] == "static_fallback"

    def test_rank_all_pathways(self):
        engine = IntegratedAssessmentEngine(country="China")
        recommendations = engine.rank_all_pathways()
        assert len(recommendations) == len(list_pathways())
        ranks = [r.rank for r in recommendations]
        assert ranks == list(range(1, len(recommendations) + 1))


class TestReportExporterDeduplication:
    """Test ReportExporter consolidated functionality."""

    def test_to_html_static_and_instance(self):
        data = {"Summary": {"Metric A": 100, "Metric B": 200}, "Notes": "Test"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path_static = os.path.join(tmpdir, "report_static.html")
            ReportExporter.to_html(data, path_static, title="Static Test")
            assert os.path.exists(path_static)
            with open(path_static, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Static Test" in content
            assert "Metric A" in content

            path_inst = os.path.join(tmpdir, "report_inst.html")
            exporter = ReportExporter()
            exporter.to_html(data, path_inst, title="Instance Test")
            assert os.path.exists(path_inst)
            with open(path_inst, "r", encoding="utf-8") as f:
                content_inst = f.read()
            assert "Instance Test" in content_inst
