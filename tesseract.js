import Tesseract from 'tesseract.js'

function Tesseracted (image_path){
    const Text = ""
    Tesseract.recognize(
    image_path,
    'eng',
    {
        logger: m => console.log(m)
    }
    ).then(
        ({data: {text}}) => {
            Text = text
        }
    )
    return Text
}
export default Tesseracted

