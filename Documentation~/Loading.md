# Importing a trained model

Before you import a trained model, you must have [exported your model to the ONNX format](Exporting.md).

## Resource loading

When you have a valid ONNX model, import it into your project; to do this, add the `.onnx` file to your project's `Assets` folder. Unity imports the model as an `NNModel` asset:

![Assets](images/Assets.png) 

![Assets](images/Inspector.png)


You can then reference this asset directly in your script as follows:

```Csharp
public NNModel modelAsset;
```

A model is an asset wrapper and is stored in binary format. You must compile it into a run-time model (of type `Model`) like this:

```Csharp
public NNModel modelAsset;
private Model m_RuntimeModel;

void Start()
{	
    m_RuntimeModel = ModelLoader.Load(modelAsset);
}    
```

## Runtime loading

You can also use the `ModelLoader` to directly load the asset from a specified path:
```Csharp
Model model = ModelLoader.Load(modelPath);
```
For more information see [Loading resources at runtime](https://docs.unity3d.com/Manual/LoadingResourcesatRuntime.html).
