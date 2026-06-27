# [ANN OOP] Save, Load và $\texttt{createFromConfig()}$ trong class $\texttt{Model}$ sử dụng $\texttt{pickle}$

## 1. Mục tiêu

Tài liệu này giải thích cách thiết kế chức năng:

* $\texttt{save()}$
* $\texttt{load()}$
* $\texttt{getConfig()}$
* $\texttt{createFromConfig()}$
* $\texttt{getParameters()}$
* $\texttt{setParameters()}$

trong một class $\texttt{Model}$ ANN tự xây dựng bằng OOP.

Thư viện sử dụng:

```python
import pickle
```

Mục tiêu cuối cùng là có thể làm được quy trình:

```python
model.save("model.pkl")
```

Sau đó ở một file khác hoặc lần chạy khác:

```python
loadedmodel = Model.load("model.pkl")
```

và dùng lại model:

```python
predictions = loadedmodel.predict(XTest)
```

---

# 2. Vì sao cần save và load model?

Sau khi train xong một ANN, ta không muốn mỗi lần chạy chương trình lại phải train lại từ đầu.

Ví dụ:

```text
Train model mất 10 phút.

Lần sau muốn dùng lại model để predict.

Không nên train lại từ đầu.
```

Ta cần lưu lại:

```text
Kiến trúc model.

Weights và biases đã học.

Cấu hình Loss, Optimizer, Accuracy nếu cần.

Trạng thái optimizer nếu muốn tiếp tục training.
```

Do đó cần các hàm:

```python
model.save(filepath)
Model.load(filepath)
```

---

# 3. Không nên chỉ lưu weights

Một ANN không chỉ có weights.

Ví dụ model:

```text
DenseLayer(2, 64)
ReLU
DenseLayer(64, 1)
Linear
```

Nếu chỉ lưu weights, khi load lại ta không biết:

```text
Model có bao nhiêu layer?

Layer nào là Dense?

Layer nào là Activation?

Dense có bao nhiêu input?

Dense có bao nhiêu neuron?

Activation là ReLU hay Linear?

Loss là gì?

Optimizer là gì?
```

Vì vậy cần tách thành hai phần:

```text
Config:
Mô tả kiến trúc và cấu hình.

Parameters:
Lưu weights và biases đã học.
```

---

# 4. Config là gì?

$\texttt{config}$ là thông tin dùng để dựng lại model.

Ví dụ:

```python
config = {
    "layers": [
        {
            "className": "DenseLayer",
            "nInputs": 2,
            "nNeurons": 64,
            "initialization": "henormal"
        },
        {
            "className": "ActivationReLU"
        },
        {
            "className": "DenseLayer",
            "nInputs": 64,
            "nNeurons": 1,
            "initialization": "xaviernormal"
        },
        {
            "className": "ActivationLinear"
        }
    ],
    "loss": {
        "className": "LossMeanSquaredError"
    },
    "optimizer": {
        "className": "OptimizerAdam",
        "learningRate": 0.001,
        "decayRate": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-7
    },
    "accuracy": {
        "className": "AccuracyRegressionTolerance",
        "tolerance": 2.0
    }
}
```

Config trả lời câu hỏi:

```text
Model này được xây dựng như thế nào?
```

---

# 5. Parameters là gì?

$\texttt{parameters}$ là các giá trị model học được sau training.

Với Dense Layer:

```text
weights

biases
```

Ví dụ:

```python
parameters = [
    {
        "weights": layer1.weights,
        "biases": layer1.biases
    },
    {
        "weights": layer2.weights,
        "biases": layer2.biases
    }
]
```

Parameters trả lời câu hỏi:

```text
Model đã học được gì?
```

---

# 6. Phân biệt $\texttt{config}$ và $\texttt{parameters}$

| Thành phần        | Ý nghĩa                     | Có thay đổi khi train không? |
| ----------------- | --------------------------- | ---------------------------- |
| $\texttt{config}$          | Kiến trúc model             | Thường không đổi             |
| $\texttt{parameters}$      | Weights và biases           | Có đổi                       |
| $\texttt{optimizerState}$ | Momentum, cache, iterations | Có đổi                       |
| $\texttt{loss}$ config     | Loại loss                   | Thường không đổi             |
| $\texttt{accuracy}$ config | Cách tính metric            | Thường không đổi             |

Tư duy:

```text
Config dùng để tạo model rỗng.

Parameters dùng để nạp lại những gì model đã học.

Optimizer state dùng để tiếp tục training mượt hơn.
```

---

# 7. Có thể pickle nguyên object $\texttt{model}$ không?

Có thể viết rất ngắn:

```python
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
```

Load lại:

```python
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
```

Cách này đơn giản nhưng có nhược điểm:

```text
Phụ thuộc mạnh vào cấu trúc class hiện tại.

Nếu sửa tên class, sửa module, sửa code OOP, file pickle cũ dễ lỗi.

Khó kiểm soát phần nào được lưu.

Có thể lưu cả những thứ không cần như inputs, output, dweights, dinputs.
```

Vì vậy cách nên dùng khi học OOP là:

```text
Không pickle trực tiếp toàn bộ model.

Thay vào đó pickle một dictionary gồm config và parameters.
```

---

# 8. Cấu trúc file pickle nên lưu

Nên lưu dạng:

```python
data = {
    "config": self.getConfig(),
    "parameters": self.getParameters(),
    "optimizerState": self.getOptimizerState()
}
```

Sau đó:

```python
with open(filepath, "wb") as file:
    pickle.dump(data, file)
```

Khi load:

```python
with open(filepath, "rb") as file:
    data = pickle.load(file)

model = Model.createFromConfig(data["config"])
model.setParameters(data["parameters"])
model.setOptimizerState(data["optimizerState"])
```

---

# 9. Cảnh báo quan trọng về $\texttt{pickle}$

$\texttt{pickle}$ không an toàn với file không tin cậy.

Không nên load file pickle lấy từ nguồn lạ:

```python
model = Model.load("unknownfile.pkl")
```

Lý do:

```text
pickle có thể thực thi code trong quá trình load.

Chỉ load file pickle do chính mình tạo hoặc từ nguồn tin cậy.
```

Trong project học tập cá nhân, dùng $\texttt{pickle}$ là ổn.

Trong production, nên cân nhắc format an toàn hơn.

---

# 10. Thiết kế OOP tổng quát

Ta muốn mỗi class tự biết cách mô tả chính nó.

Ví dụ:

```text
DenseLayer có getConfig().

ActivationReLU có getConfig().

OptimizerAdam có getConfig().

Loss có getConfig().

Accuracy có getConfig().
```

Sau đó $\texttt{Model}$ chỉ cần gọi:

```python
layer.getConfig()
```

cho từng layer.

Tư duy:

```text
Model không cần biết chi tiết từng class.

Mỗi component tự mô tả config của chính nó.
```

---

# 11. $\texttt{getConfig()}$ trong $\texttt{DenseLayer}$

Ví dụ $\texttt{DenseLayer}$:

```python
class DenseLayer:
    def init(
        self,
        nInputs,
        nNeurons,
        initialization="smallrandom"
    ):
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        self.initialization = initialization

        self.weights = (
            0.01
            * np.random.randn(
                nInputs,
                nNeurons
            )
        )

        self.biases = np.zeros(
            (1, nNeurons)
        )

    def getConfig(self):
        return {
            "className": "DenseLayer",
            "nInputs": self.nInputs,
            "nNeurons": self.nNeurons,
            "initialization": self.initialization
        }
```

Khi gọi:

```python
dense.getConfig()
```

ta nhận:

```python
{
    "className": "DenseLayer",
    "nInputs": 2,
    "nNeurons": 64,
    "initialization": "smallrandom"
}
```

---

# 12. $\texttt{getParameters()}$ trong $\texttt{DenseLayer}$

Dense Layer có weights và biases.

Do đó cần:

```python
def getParameters(self):
    return {
        "weights": self.weights,
        "biases": self.biases
    }
```

Và để load lại:

```python
def setParameters(self, parameters):
    self.weights = parameters["weights"]
    self.biases = parameters["biases"]
```

Full code:

```python
class DenseLayer:
    def init(
        self,
        nInputs,
        nNeurons,
        initialization="smallrandom"
    ):
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        self.initialization = initialization

        self.weights = (
            0.01
            * np.random.randn(
                nInputs,
                nNeurons
            )
        )

        self.biases = np.zeros(
            (1, nNeurons)
        )

    def getConfig(self):
        return {
            "className": "DenseLayer",
            "nInputs": self.nInputs,
            "nNeurons": self.nNeurons,
            "initialization": self.initialization
        }

    def getParameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }

    def setParameters(
        self,
        parameters
    ):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]
```

---

# 13. $\texttt{getConfig()}$ trong Activation

Activation thường không có weights.

Ví dụ ReLU:

```python
class ActivationReLU:
    def getConfig(self):
        return {
            "className": "ActivationReLU"
        }
```

Linear:

```python
class ActivationLinear:
    def getConfig(self):
        return {
            "className": "ActivationLinear"
        }
```

Vì Activation không có weights nên không cần $\texttt{getParameters()}$.

Nếu muốn thống nhất interface, có thể viết:

```python
def getParameters(self):
    return None

def setParameters(self, parameters):
    pass
```

Nhưng cách đơn giản hơn là $\texttt{Model}$ chỉ lấy parameters từ layer có $\texttt{weights}$.

---

# 14. $\texttt{getConfig()}$ trong Loss

Ví dụ MSE:

```python
class LossMeanSquaredError:
    def getConfig(self):
        return {
            "className": "LossMeanSquaredError"
        }
```

Nếu Loss có tham số, ví dụ L1 hoặc L2 coefficient, cần lưu thêm.

Ví dụ:

```python
class LossMSEWithRegularization:
    def init(
        self,
        lambdaL2=0.0
    ):
        self.lambdaL2 = lambdaL2

    def getConfig(self):
        return {
            "className": "LossMSEWithRegularization",
            "lambdaL2": self.lambdaL2
        }
```

---

# 15. $\texttt{getConfig()}$ trong Optimizer

Optimizer có nhiều hyperparameter.

Ví dụ Adam:

```python
class OptimizerAdam:
    def init(
        self,
        learningRate=0.001,
        decayRate=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7
    ):
        self.initialLearningRate = learningRate
        self.currentLearningRate = learningRate
        self.decayRate = decayRate

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.iterations = 0

    def getConfig(self):
        return {
            "className": "OptimizerAdam",
            "learningRate": self.initialLearningRate,
            "decayRate": self.decayRate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }
```

Lưu ý:

```text
getConfig() lưu hyperparameters.

Không nhất thiết lưu trạng thái momentum/cache ở đây.
```

Trạng thái momentum/cache nên tách thành:

```python
getOptimizerState()
```

---

# 16. $\texttt{getConfig()}$ trong Accuracy

Ví dụ Regression Tolerance Accuracy:

```python
class AccuracyRegressionTolerance:
    def init(
        self,
        tolerance
    ):
        self.tolerance = tolerance

    def getConfig(self):
        return {
            "className": "AccuracyRegressionTolerance",
            "tolerance": self.tolerance
        }
```

Categorical Accuracy không có tham số:

```python
class AccuracyCategorical:
    def getConfig(self):
        return {
            "className": "AccuracyCategorical"
        }
```

---

# 17. Vì sao cần $\texttt{createFromConfig()}$?

$\texttt{createFromConfig()}$ dùng để dựng lại model từ config.

Ví dụ config:

```python
{
    "layers": [
        {
            "className": "DenseLayer",
            "nInputs": 2,
            "nNeurons": 64
        },
        {
            "className": "ActivationReLU"
        }
    ]
}
```

Ta muốn tự động tạo:

```python
model = Model()

model.add(
    DenseLayer(
        nInputs=2,
        nNeurons=64
    )
)

model.add(
    ActivationReLU()
)
```

Nhưng không viết thủ công.

Do đó cần:

```python
Model.createFromConfig(config)
```

---

# 18. Registry là gì?

Để tạo object từ tên class, ta cần một dictionary ánh xạ tên class sang class thật.

Ví dụ:

```python
CLASSREGISTRY = {
    "DenseLayer": DenseLayer,
    "ActivationReLU": ActivationReLU,
    "ActivationLinear": ActivationLinear,
    "LossMeanSquaredError": LossMeanSquaredError,
    "OptimizerAdam": OptimizerAdam,
    "AccuracyRegressionTolerance": AccuracyRegressionTolerance
}
```

Khi thấy:

```python
"className": "DenseLayer"
```

ta lấy class thật:

```python
classtype = CLASSREGISTRY["DenseLayer"]
```

rồi tạo object:

```python
layer = classtype(
    nInputs=2,
    nNeurons=64
)
```

---

# 19. Hàm tạo object từ config

Ta có thể viết helper function:

```python
def createobjectfromconfig(
    config
):
    className = config["className"]

    if className not in CLASSREGISTRY:
        raise ValueError(
            f"Không tìm thấy class: {className}"
        )

    classtype = CLASSREGISTRY[className]

    kwargs = config.copy()
    kwargs.pop("className")

    return classtype(**kwargs)
```

Ví dụ:

```python
config = {
    "className": "DenseLayer",
    "nInputs": 2,
    "nNeurons": 64
}

layer = createobjectfromconfig(config)
```

Tương đương:

```python
layer = DenseLayer(
    nInputs=2,
    nNeurons=64
)
```

---

# 20. $\texttt{Model.getConfig()}$

Trong $\texttt{Model}$, ta lưu config như sau:

```python
def getConfig(self):
    config = {
        "layers": [],
        "loss": None,
        "optimizer": None,
        "accuracy": None
    }

    for layer in self.layers:
        config["layers"].append(
            layer.getConfig()
        )

    if self.loss is not None:
        config["loss"] = self.loss.getConfig()

    if self.optimizer is not None:
        config["optimizer"] = (
            self.optimizer.getConfig()
        )

    if self.accuracy is not None:
        config["accuracy"] = (
            self.accuracy.getConfig()
        )

    return config
```

Kết quả là một dictionary mô tả toàn bộ model.

---

# 21. $\texttt{Model.createFromConfig()}$

$\texttt{createFromConfig()}$ nên là $\texttt{classmethod}$.

Lý do:

```text
Nó tạo ra một object Model mới.

Nó thuộc về class Model, không cần object Model có sẵn.
```

Code:

```python
@classmethod
def createFromConfig(
    cls,
    config
):
    model = cls()

    for layerConfig in config["layers"]:
        layer = createobjectfromconfig(
            layerConfig
        )

        model.add(layer)

    lossConfig = config.get("loss")

    if lossConfig is not None:
        loss = createobjectfromconfig(
            lossConfig
        )
    else:
        loss = None

    optimizerConfig = config.get(
        "optimizer"
    )

    if optimizerConfig is not None:
        optimizer = createobjectfromconfig(
            optimizerConfig
        )
    else:
        optimizer = None

    accuracyConfig = config.get(
        "accuracy"
    )

    if accuracyConfig is not None:
        accuracy = createobjectfromconfig(
            accuracyConfig
        )
    else:
        accuracy = None

    model.set(
        loss=loss,
        optimizer=optimizer,
        accuracy=accuracy
    )

    model.finalize()

    return model
```

Tư duy:

```text
Đọc config.

Tạo layer theo đúng thứ tự.

Tạo loss.

Tạo optimizer.

Tạo accuracy.

Set vào model.

Finalize model.

Trả về model mới.
```

---

# 22. $\texttt{Model.getParameters()}$

Model chỉ nên lấy parameters từ các trainable layers.

Trainable layers thường là layer có $\texttt{weights}$.

```python
def getParameters(self):
    parameters = []

    for layer in self.trainableLayers:
        parameters.append(
            layer.getParameters()
        )

    return parameters
```

Ví dụ model có 2 Dense Layer:

```python
parameters = [
    {
        "weights": dense1.weights,
        "biases": dense1.biases
    },
    {
        "weights": dense2.weights,
        "biases": dense2.biases
    }
]
```

---

# 23. $\texttt{Model.setParameters()}$

Khi load, ta cần nạp lại parameters vào trainable layers.

```python
def setParameters(
    self,
    parameters
):
    if len(parameters) != len(self.trainableLayers):
        raise ValueError(
            "Số lượng parameter không khớp với số trainable layers."
        )

    for layer, layerParameters in zip(
        self.trainableLayers,
        parameters
    ):
        layer.setParameters(
            layerParameters
        )
```

Quan trọng:

```text
Phải gọi finalize() trước khi setParameters().

Vì finalize() tạo self.trainableLayers.
```

---

# 24. Có cần lưu optimizer state không?

Nếu chỉ muốn predict sau khi load:

```text
Không bắt buộc lưu optimizer state.
```

Nếu muốn tiếp tục training từ đúng trạng thái cũ:

```text
Nên lưu optimizer state.
```

Ví dụ Adam có:

```text
iterations

currentLearningRate

momentums của từng layer

cache của từng layer
```

Nếu không lưu optimizer state, model vẫn có weights đúng, nhưng Adam sẽ khởi động lại từ đầu về mặt momentum/cache.

Điều này không sai, nhưng quá trình train tiếp có thể hơi khác.

---

# 25. Cách lưu optimizer state đơn giản

Vì Adam thường lưu momentum/cache trong từng layer:

```text
layer.weightMomentums

layer.biasMomentums

layer.weightCache

layer.biasCache
```

Ta có thể lưu chúng cùng parameters.

Ví dụ trong $\texttt{DenseLayer.getParameters()}$:

```python
def getParameters(self):
    parameters = {
        "weights": self.weights,
        "biases": self.biases
    }

    if hasattr(self, "weightMomentums"):
        parameters["weightMomentums"] = (
            self.weightMomentums
        )

        parameters["biasMomentums"] = (
            self.biasMomentums
        )

    if hasattr(self, "weightCache"):
        parameters["weightCache"] = (
            self.weightCache
        )

        parameters["biasCache"] = (
            self.biasCache
        )

    return parameters
```

Và khi set:

```python
def setParameters(
    self,
    parameters
):
    self.weights = parameters["weights"]
    self.biases = parameters["biases"]

    if "weightMomentums" in parameters:
        self.weightMomentums = (
            parameters["weightMomentums"]
        )

        self.biasMomentums = (
            parameters["biasMomentums"]
        )

    if "weightCache" in parameters:
        self.weightCache = (
            parameters["weightCache"]
        )

        self.biasCache = (
            parameters["biasCache"]
        )
```

---

# 26. Lưu $\texttt{optimizer.iterations}$

Ngoài momentum/cache trong layer, cần lưu số lần update của optimizer.

Trong $\texttt{OptimizerAdam}$:

```python
def getstate(self):
    return {
        "iterations": self.iterations,
        "currentLearningRate": (
            self.currentLearningRate
        )
    }

def setstate(
    self,
    state
):
    if state is None:
        return

    self.iterations = state.get(
        "iterations",
        0
    )

    self.currentLearningRate = state.get(
        "currentLearningRate",
        self.initialLearningRate
    )
```

Nếu optimizer không có state đặc biệt, có thể viết:

```python
def getstate(self):
    return {}

def setstate(self, state):
    pass
```

---

# 27. $\texttt{Model.getOptimizerState()}$

Trong $\texttt{Model}$:

```python
def getOptimizerState(self):
    if self.optimizer is None:
        return None

    if hasattr(self.optimizer, "getstate"):
        return self.optimizer.getstate()

    return None
```

---

# 28. $\texttt{Model.setOptimizerState()}$

Trong $\texttt{Model}$:

```python
def setOptimizerState(
    self,
    state
):
    if self.optimizer is None:
        return

    if hasattr(self.optimizer, "setstate"):
        self.optimizer.setstate(state)
```

---

# 29. $\texttt{Model.save()}$

$\texttt{save()}$ dùng pickle để lưu dictionary.

```python
def save(
    self,
    filepath
):
    data = {
        "config": self.getConfig(),
        "parameters": self.getParameters(),
        "optimizerState": self.getOptimizerState()
    }

    with open(filepath, "wb") as file:
        pickle.dump(
            data,
            file
        )
```

Cách dùng:

```python
model.save("quadraticmodel.pkl")
```

File $\texttt{.pkl}$ sẽ chứa:

```text
config

parameters

optimizerState
```

---

# 30. $\texttt{Model.load()}$

$\texttt{load()}$ nên là $\texttt{classmethod}$.

```python
@classmethod
def load(
    cls,
    filepath
):
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    model = cls.createFromConfig(
        data["config"]
    )

    model.setParameters(
        data["parameters"]
    )

    model.setOptimizerState(
        data.get("optimizerState")
    )

    return model
```

Cách dùng:

```python
loadedmodel = Model.load(
    "quadraticmodel.pkl"
)
```

Sau đó:

```python
yPred = loadedmodel.predict(XTest)
```

---

# 31. Full code class $\texttt{Model}$

```python
import pickle

class Model:
    def init(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.trainableLayers = []

    def add(
        self,
        layer
    ):
        self.layers.append(layer)

    def set(
        self,
        loss=None,
        optimizer=None,
        accuracy=None
    ):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.trainableLayers = []

        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.trainableLayers.append(
                    layer
                )

    def getConfig(self):
        config = {
            "layers": [],
            "loss": None,
            "optimizer": None,
            "accuracy": None
        }

        for layer in self.layers:
            config["layers"].append(
                layer.getConfig()
            )

        if self.loss is not None:
            config["loss"] = (
                self.loss.getConfig()
            )

        if self.optimizer is not None:
            config["optimizer"] = (
                self.optimizer.getConfig()
            )

        if self.accuracy is not None:
            config["accuracy"] = (
                self.accuracy.getConfig()
            )

        return config

    @classmethod
    def createFromConfig(
        cls,
        config
    ):
        model = cls()

        for layerConfig in config["layers"]:
            layer = createobjectfromconfig(
                layerConfig
            )

            model.add(layer)

        loss = None
        optimizer = None
        accuracy = None

        if config.get("loss") is not None:
            loss = createobjectfromconfig(
                config["loss"]
            )

        if config.get("optimizer") is not None:
            optimizer = createobjectfromconfig(
                config["optimizer"]
            )

        if config.get("accuracy") is not None:
            accuracy = createobjectfromconfig(
                config["accuracy"]
            )

        model.set(
            loss=loss,
            optimizer=optimizer,
            accuracy=accuracy
        )

        model.finalize()

        return model

    def getParameters(self):
        parameters = []

        for layer in self.trainableLayers:
            parameters.append(
                layer.getParameters()
            )

        return parameters

    def setParameters(
        self,
        parameters
    ):
        if len(parameters) != len(
            self.trainableLayers
        ):
            raise ValueError(
                "Số lượng parameter không khớp với số trainable layers."
            )

        for layer, layerParameters in zip(
            self.trainableLayers,
            parameters
        ):
            layer.setParameters(
                layerParameters
            )

    def getOptimizerState(self):
        if self.optimizer is None:
            return None

        if hasattr(
            self.optimizer,
            "getstate"
        ):
            return self.optimizer.getstate()

        return None

    def setOptimizerState(
        self,
        state
    ):
        if self.optimizer is None:
            return

        if hasattr(
            self.optimizer,
            "setstate"
        ):
            self.optimizer.setstate(state)

    def save(
        self,
        filepath
    ):
        data = {
            "config": self.getConfig(),
            "parameters": self.getParameters(),
            "optimizerState": self.getOptimizerState()
        }

        with open(filepath, "wb") as file:
            pickle.dump(
                data,
                file
            )

    @classmethod
    def load(
        cls,
        filepath
    ):
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        model = cls.createFromConfig(
            data["config"]
        )

        model.setParameters(
            data["parameters"]
        )

        model.setOptimizerState(
            data.get("optimizerState")
        )

        return model
```

---

# 32. Full code registry

```python
CLASSREGISTRY = {
    "DenseLayer": DenseLayer,
    "ActivationReLU": ActivationReLU,
    "ActivationLinear": ActivationLinear,
    "LossMeanSquaredError": LossMeanSquaredError,
    "OptimizerAdam": OptimizerAdam,
    "AccuracyRegressionTolerance": AccuracyRegressionTolerance
}

def createobjectfromconfig(
    config
):
    className = config["className"]

    if className not in CLASSREGISTRY:
        raise ValueError(
            f"Không tìm thấy class: {className}"
        )

    classtype = CLASSREGISTRY[className]

    kwargs = config.copy()
    kwargs.pop("className")

    return classtype(**kwargs)
```

Lưu ý:

```text
Mỗi khi tạo thêm class mới, cần thêm class đó vào CLASSREGISTRY.
```

Ví dụ thêm Softmax:

```python
CLASSREGISTRY["ActivationSoftmax"] = (
    ActivationSoftmax
)
```

---

# 33. Ví dụ sử dụng với Quadratic Regression

Bài toán:

$$y=ax^2+bx+c$$

Tạo feature:

```python
XQuad = np.c[
    X ** 2,
    X
]
```

Tạo model:

```python
model = Model()

model.add(
    DenseLayer(
        nInputs=2,
        nNeurons=1
    )
)

model.add(
    ActivationLinear()
)

model.set(
    loss=LossMeanSquaredError(),
    optimizer=OptimizerAdam(
        learningRate=0.001
    ),
    accuracy=AccuracyRegressionTolerance(
        tolerance=2.0
    )
)

model.finalize()
```

Train:

```python
model.train(
    XQuad,
    y,
    epochs=10000
)
```

Lưu model:

```python
model.save("quadraticmodel.pkl")
```

Load model:

```python
loadedmodel = Model.load(
    "quadraticmodel.pkl"
)
```

Predict:

```python
yPred = loadedmodel.predict(
    XQuad
)
```

Lấy hệ số:

```python
dense = loadedmodel.layers[0]

a = dense.weights[0, 0]
b = dense.weights[1, 0]
c = dense.biases[0, 0]

print("a =", a)
print("b =", b)
print("c =", c)
```

---

# 34. Ví dụ file lưu ra có gì?

Sau khi gọi:

```python
model.save("model.pkl")
```

Dữ liệu trong file tương đương:

```python
{
    "config": {
        "layers": [
            {
                "className": "DenseLayer",
                "nInputs": 2,
                "nNeurons": 1,
                "initialization": "smallrandom"
            },
            {
                "className": "ActivationLinear"
            }
        ],
        "loss": {
            "className": "LossMeanSquaredError"
        },
        "optimizer": {
            "className": "OptimizerAdam",
            "learningRate": 0.001,
            "decayRate": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-7
        },
        "accuracy": {
            "className": "AccuracyRegressionTolerance",
            "tolerance": 2.0
        }
    },
    "parameters": [
        {
            "weights": "... numpy array ...",
            "biases": "... numpy array ..."
        }
    ],
    "optimizerState": {
        "iterations": 10000,
        "currentLearningRate": 0.001
    }
}
```

---

# 35. Vì sao không lưu $\texttt{inputs}$, $\texttt{output}$, $\texttt{dweights}$, $\texttt{dinputs}$?

Trong quá trình forward và backward, layer thường có:

```text
inputs

output

dweights

dbiases

dinputs
```

Các biến này chỉ phục vụ cho một lần forward/backward hiện tại.

Không nên lưu chúng vào file model.

Lý do:

```text
Làm file lớn hơn.

Không cần cho prediction.

Dễ gây lỗi nếu shape dữ liệu thay đổi.

Không phải bản chất model đã học.
```

Nên lưu:

```text
weights

biases

optimizer state nếu cần train tiếp
```

---

# 36. Save để predict và save để train tiếp khác nhau thế nào?

## Save để predict

Cần lưu:

```text
config

parameters
```

Không nhất thiết cần:

```text
optimizerState
```

## Save để train tiếp

Nên lưu:

```text
config

parameters

optimizerState

momentums/cache nếu dùng Adam hoặc Momentum
```

Nếu không lưu optimizer state, vẫn train tiếp được, nhưng optimizer giống như bắt đầu lại.

---

# 37. Checklist thiết kế save/load cho Model

```text
Mỗi component có getConfig().

DenseLayer có getParameters() và setParameters().

Optimizer có getConfig().

Optimizer có getstate() và setstate() nếu cần.

Model có getConfig().

Model có createFromConfig().

Model có getParameters().

Model có setParameters().

Model có save().

Model có load().

Có CLASSREGISTRY để ánh xạ tên class sang class thật.

Không load pickle từ nguồn không tin cậy.
```

---

# 38. Những lỗi thường gặp

## Lỗi 1: Quên gọi $\texttt{finalize()}$ trước khi $\texttt{setParameters()}$

Nếu chưa gọi:

```python
model.finalize()
```

thì:

```python
self.trainableLayers
```

có thể rỗng.

Khi đó không biết gán weights vào layer nào.

---

## Lỗi 2: Không lưu config

Nếu chỉ lưu weights:

```text
Không dựng lại được kiến trúc model.
```

---

## Lỗi 3: Không thêm class mới vào registry

Nếu config có:

```python
{
    "className": "ActivationSoftmax"
}
```

nhưng registry không có:

```python
"ActivationSoftmax": ActivationSoftmax
```

thì load sẽ lỗi.

---

## Lỗi 4: Lưu toàn bộ object quá sớm

Dùng:

```python
pickle.dump(model, file)
```

rất nhanh, nhưng khó kiểm soát.

Khi đang học OOP, nên tự viết:

```python
save()
load()
getConfig()
createFromConfig()
```

để hiểu hệ thống hoạt động.

---

## Lỗi 5: Load pickle từ nguồn lạ

Không nên:

```python
Model.load("filenguonla.pkl")
```

vì pickle không an toàn với nguồn không tin cậy.

---

# 39. Tóm tắt tư duy

Muốn save/load một ANN OOP tốt, cần tách:

```text
Kiến trúc model.

Tham số đã học.

Trạng thái optimizer.
```

Tương ứng:

```text
config

parameters

optimizerState
```

Các hàm quan trọng:

```text
getConfig():
Mô tả object.

createFromConfig():
Tạo lại object từ mô tả.

getParameters():
Lấy weights, biases.

setParameters():
Nạp lại weights, biases.

save():
pickle.dump dictionary.

load():
pickle.load dictionary rồi dựng lại model.
```

Tư duy cuối cùng:

```text
Model.save() không chỉ là lưu weights.

Model.load() không chỉ là đọc file.

Save/load tốt phải giúp model được dựng lại đúng kiến trúc và đúng tham số đã học.

createFromConfig() là cầu nối giữa config dạng dictionary và object Python thật.
```
