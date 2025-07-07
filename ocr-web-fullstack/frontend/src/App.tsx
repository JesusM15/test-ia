import { useState, useRef } from 'react'
import './App.css'

function App() {
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedImageURL, setSelectedImageURL] = useState<string | null>(null);
  const [ selectedFile, setSelectedFile ] = useState<File | null>(null);
  const [ showError, setShowError ] = useState<boolean | null>(false);
  const [ errorMessage, setErrorMessage ] = useState<String | null>(null);
  const [ text, setText ] = useState<String | null>(null);
  const [ isLoading, setIsLoading ] = useState<boolean>(false);

  const handleOnPress = () => {
    if(imageInputRef.current){
      imageInputRef.current.click();
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        alert("Por favor selecciona una imagen válida.");
        return;
      }

      const imageUrl = URL.createObjectURL(file);
      setSelectedFile(file);
      setSelectedImageURL(imageUrl);
    }
  };

  const handleProcessImage = async () => {
    if(!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setIsLoading(true);
      const response = await fetch("http://127.0.0.1:8000/upload-ocr", {
        method: "POST",
        body: formData
      });

      if(!response.ok){
        const error = await response.json();
        setErrorMessage(error?.detail);
        setShowError(true);
        setIsLoading(false);
        return;
      };

      setShowError(false);
      setErrorMessage(null);
      const data = await response.json();
      setText(data.text);
      setIsLoading(false); 
    } catch(error){
      console.warn("Error procesando la imagen: ", error);
    }
  }

  return (
    <main className='background'>
      <article className='container'>
        <h1 style={{color: "#fff", fontSize: "1.8em"}}>
          Procesador de imagenes a texto
        </h1>
        {selectedImageURL && (
          <img
            src={selectedImageURL}
            alt="Vista previa"
            style={{ width: '100%', maxWidth: 300, marginBottom: 16, borderRadius: 8 }}
          />
        )}
        <input type="file" hidden  ref={imageInputRef}
          onChange={handleImageChange}
          accept="image/*"
        />
        <div style={{display: "flex", justifyContent: 'space-between', gap: 12}}>
          <button className='btn-load'
          onClick={handleOnPress} disabled={isLoading}
          >
            Cargar imagen
          </button>
          <button className='btn-process' onClick={handleProcessImage}>
            Procesar
          </button>
        </div>
        {isLoading && (
  <p style={{ color: "#ccc", marginTop: 12 }}>Procesando imagen, espera un momento...</p>
)}
       {(text || showError) && !isLoading && <div className='text-container'>
          <p style={{
            color: "#fff"
          }}>
            {
              text ?? errorMessage
            }
          </p>
        </div>}
      </article>
    </main>
  )
}

export default App
