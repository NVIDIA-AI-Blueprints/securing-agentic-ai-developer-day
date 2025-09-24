# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NeMo Guardrails generator."""

from contextlib import redirect_stderr
import io
from typing import List, Union

from garak import _config
from garak.generators.base import Generator
from .config import init_main_llm


class Guardrails(Generator):
    """Custom Generator wrapper for NeMo Guardrails."""

    supports_multiple_generations = False
    generator_family_name = "Guardrails"

    def __init__(self, name="", config_root=_config):
        # another class that may need to skip testing due to non required dependency
        try:
            from nemoguardrails import RailsConfig, LLMRails
            from nemoguardrails.logging.verbose import set_verbose
        except ImportError as e:
            raise NameError(
                "You must first install NeMo Guardrails using `pip install nemoguardrails`."
            ) from e

        self.name = name
        self._load_config(config_root)
        self.fullname = f"Guardrails {self.name}"

        # Currently, we use the model_name as the path to the config
        with redirect_stderr(io.StringIO()) as f:  # quieten the tqdm
            config = RailsConfig.from_path(self.name)
            init_main_llm(config)
            self.rails = LLMRails(config=config)

        super().__init__(self.name, config_root=config_root)

    def _call_model(
        self, prompt: str, generations_this_call: int = 1
    ) -> List[Union[str, None]]:
        with redirect_stderr(io.StringIO()) as f:  # quieten the tqdm
            result = self.rails.generate(prompt)

        return [result]


DEFAULT_CLASS = "NeMoGuardrails"