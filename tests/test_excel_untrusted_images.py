"""
An uploaded workbook's pictures never reach Pillow, and what is extracted is unchanged.

openpyxl's full load (every workbook up to 25 MB) opened each embedded picture with
PIL.Image.open to read its size: the one path by which an upload reached Pillow's
image parsers, the subject of most published Pillow advisories (Decision 104).
"""
from __future__ import annotations

import dataclasses

import openpyxl
import openpyxl.reader.drawings as drawings
import PIL.Image
import pytest
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as SheetImage

import src.data_processing.excel_processor as excel_processor
from src.data_processing.excel_processor import ExcelProcessor


@pytest.fixture
def workbook_with_pictures(tmp_path):
    png = tmp_path / "logo.png"
    PIL.Image.new("RGB", (8, 8), (200, 10, 10)).save(png)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["USD in thousands", "FY2022", "FY2023"])
    ws.append(["Revenue", 1200, 1350.5])
    ws.append(["EBITDA", 300, 0.25])
    ws["C3"].number_format = "0.0%"
    ws.merge_cells("A5:C5")
    ws["A5"] = "Management accounts, unaudited"
    ws.add_image(SheetImage(str(png)), "E2")
    chart = BarChart()
    chart.add_data(Reference(ws, min_col=2, min_row=1, max_col=3, max_row=3), titles_from_data=True)
    ws.add_chart(chart, "E10")
    second = wb.create_sheet("Pictures only")
    second.add_image(SheetImage(str(png)), "B2")
    second["A1"] = "Site photo"
    path = tmp_path / "pictures.xlsx"
    wb.save(path)
    return str(path)


@pytest.fixture
def pillow_opens(monkeypatch):
    calls = []
    real_open = PIL.Image.open

    def spy(fp, *args, **kwargs):
        calls.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(PIL.Image, "open", spy)
    return calls


def _extract(path):
    return [dataclasses.asdict(s) for s in ExcelProcessor().process(path, doc_id="doc-pics")]


def test_a_workbook_with_pictures_is_read_without_opening_them(workbook_with_pictures, pillow_opens):
    sheets = _extract(workbook_with_pictures)
    assert sheets, "the workbook's table was not extracted"
    assert pillow_opens == []


def test_the_extraction_is_the_same_as_when_pictures_were_opened(workbook_with_pictures, pillow_opens, monkeypatch):
    guarded = _extract(workbook_with_pictures)
    assert pillow_opens == []

    # The old behaviour: pictures opened with Pillow during the load.
    monkeypatch.setattr(excel_processor, "_skip_embedded_images", lambda: None)
    monkeypatch.setattr(drawings, "PILImage", PIL.Image)
    unguarded = _extract(workbook_with_pictures)
    assert len(pillow_opens) == 2, "the fixture must exercise openpyxl's picture loading"
    assert guarded == unguarded
