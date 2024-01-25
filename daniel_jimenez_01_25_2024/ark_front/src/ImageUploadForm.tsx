import React, { useState,useRef } from 'react';
import { Button, TextField } from '@mui/material';
import BoundingBoxes from './BoundingBox';

const ImageUploadForm: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [emotionData, setEmotionData] = useState<{ emotion: string; box: number[]}|null>(null);
    const [imageURL, setImageURL] = useState<string | null>(null);
    const imageRef = useRef<HTMLImageElement>(null);


    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const file = event.target.files[0];
            setSelectedFile(file);
            const reader = new FileReader();
            
            reader.onloadend = () => {
                setImageURL(reader.result as string);
            };
    
            reader.readAsDataURL(file);
        }
        setEmotionData(null)

    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!selectedFile) {
            alert('Please select a file first!');
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('http://localhost:5000/detect-emotion', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const responseBody = await response.json();
            // console.log(responseBody)
            if (responseBody.emotion) {
                setEmotionData(responseBody);
            } else {
                alert('No emotions detected.');
            }

        } catch (error) {
            alert('Failed to upload image.');
            console.error(error);
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{width:'30%', display:'flex', flexDirection: 'column'}}>
             <div style={{ position: 'relative' }}>
                {imageURL && <img 
                src={imageURL} 
                ref={imageRef}
                alt="Uploaded" 
                style={{maxWidth:'100%', display: 'block'}}
                />}
                {emotionData && <BoundingBoxes boxes={emotionData} imageRef={imageRef} />}
             </div>
             
            <TextField
                type="file"
                onChange={handleFileChange}
                variant="outlined"
                margin="normal"
                fullWidth
            />
            <Button type="submit" variant="contained" color="primary">
                get prediction
            </Button>
        </form>
    );
};

export default ImageUploadForm;
