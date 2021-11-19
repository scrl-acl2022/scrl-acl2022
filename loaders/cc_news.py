import os
import json
import datasets
from pathlib import Path


_DESCRIPTION = "CommonCrawl-News Dataset for Sentence Compression"

_ID = "id"
_TITLE = "title"
_DOCUMENT = "document"
_SUMMARY = "summary"



class CommonCrawl(datasets.GeneratorBasedBuilder):
    """Commoncrawl Dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _ID: datasets.Value("string"),
                    _TITLE: datasets.Value("string"),
                    _DOCUMENT: datasets.Value("string"),
                    _SUMMARY: datasets.Value("string"),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager._data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(data_dir, "train.jsonl"), "name": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(data_dir, "val.jsonl"), "name": "val"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(data_dir, "test.jsonl"), "name": "test"}
            ),
        ]

    def _generate_examples(self, path=None, name=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = json.loads(line)
                id = f"{name}-{i}"
                item = {
                    _ID: id,
                    _DOCUMENT: x["sentence"],
                    _TITLE: x["title"],                    
                    _SUMMARY: json.dumps(x["description"])
                }
                yield id, item
