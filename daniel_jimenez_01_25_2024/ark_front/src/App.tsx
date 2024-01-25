
import ImageUploadForm from "./ImageUploadForm"
import darkTheme from './theme';
import { ThemeProvider } from '@mui/material';

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <div style={{
        width:'100%', 
        display: 'flex',
        justifyContent:'center',
        height: '100vh',
        alignItems:'center',
        boxSizing: 'border-box',
        }}>
        <ImageUploadForm/>
      </div>      
    </ThemeProvider>
  )
}

export default App
