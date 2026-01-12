from src.metadata_processor import MetadataProcessor


def write_csv(tmp_path, content: str) -> str:
    p = tmp_path / "test_meta_new.csv"
    p.write_text(content)
    return str(p)


def test_metadata_standard_flow(tmp_path):
    csv = """File,N,left
KDR_001_Someplace_1901.png,52,13.4
KDR_002_Other_1950.png,52.1,13.5
"""
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    df = proc.load_csv()
    assert 'longName' in df.columns
    proc.add_temporal_metadata()
    assert list(df['year']) == [1901, 1950]


def test_resolve_image_paths(tmp_path):
    csv = """shortName,longName,N,left
IMG1,KDR_001_IMG1_1901.png,52,13
,KDR_002_IMG2_1950.png,52,14
"""
    imgdir = tmp_path / 'imgdir'
    imgdir.mkdir()
    (imgdir / 'IMG1.png').write_text('x')
    (imgdir / 'kdr_002_img2_1950.png').write_text('x')
    path = write_csv(tmp_path, csv)
    proc = MetadataProcessor(path)
    proc.load_csv()
    df = proc.resolve_image_paths(str(imgdir), prefer_shortname=True)
    assert df['image_filename'].iloc[0] == 'IMG1.png'
    assert df['image_path'].iloc[1] is not None
