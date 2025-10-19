import json
import os
import datasets

_DESCRIPTION = """\
LooGLE is a comprehensive evaluation benchmark for LLM long context understanding which contains up-to-date (all after 2022) and extreme long realistic documents (over 24k tokens per document, many of which are exceeding 100k words) from diverse domains and categories.
"""

_HOMEPAGE = """\
https://github.com/bigai-nlco/LooGLE
"""

_URL = r"data.zip"


task_list = ["shortdep_qa","longdep_qa","longdep_summarization","shortdep_cloze"]


class LooGLEConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class LooGLE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LooGLEConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"),  
                "title": datasets.Value("string"), 
                "qa_pairs": datasets.Value("string"), 
                "output": datasets.Value("string")
                }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, f"{task_name}.json"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                key = f"{self.config.name}-{idx}"
                item = json.loads(line)
                yield key, {
                    "input": item["input"],
                    "title": item["title"],
                    "qa_pairs": item["qa_pairs"],
                    "output": item["output"]
                }