const BoundingBoxes: React.FC<{ boxes: { emotion: string; box: number[] }; imageRef: React.RefObject<HTMLImageElement> }> =
 ({ boxes, imageRef }) => {
    
    const calculateScaledBox = (box: number[]) => {
        console.log("calculateScaledBox called",imageRef.current);
        if (!imageRef.current) return { left: 0, top: 0, width: 0, height: 0 };
   
        const scaleX = imageRef.current.clientWidth / imageRef.current.naturalWidth;
        const scaleY = imageRef.current.clientHeight / imageRef.current.naturalHeight;

        return {
            left: box[0] * scaleX,
            top: box[1] * scaleY,
            width: box[2] * scaleX,
            height: box[3] * scaleY,
        };
    };
    return (
        <div style={{ position: 'absolute', top: 0, left: 0 , border:'2px solid red'}}>
                <div  style={{
                    border: '2px solid red',
                    position: 'absolute',
                    ...calculateScaledBox(boxes.box)
                }}>
                    <span style={{
                        position: 'absolute',
                        bottom: '100%',
                        left: '0',     
                        color: 'white',
                        backgroundColor: 'red',
                        padding: '2px',
                    }}>
                        {boxes.emotion}
                    </span>
                </div>
        </div>
    );
};

export default BoundingBoxes;