from transformers import Pipeline


class MeTTSEvaluation(Pipeline):
    # def _sanitize_parameters(self, **kwargs):
    #     preprocess_kwargs = {}
    #     if "maybe_arg" in kwargs:
    #         preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
    #     return preprocess_kwargs, {}, {}

    def preprocess(self, inputs):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class