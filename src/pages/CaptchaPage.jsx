import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function TextCaptcha() {
    const navigate = useNavigate();
    const [captchaQuestion, setCaptchaQuestion] = useState("");
    const [userAnswer, setUserAnswer] = useState("");
    const [isCorrect, setIsCorrect] = useState(null);
    const [number1, setNumber1] = useState(0);
    const [number2, setNumber2] = useState(0);

    useEffect(() => {
        generateCaptcha();
    }, []);

    const generateCaptcha = () => {
        const num1 = Math.floor(Math.random() * 10) + 1;
        const num2 = Math.floor(Math.random() * 10) + 1;
        setNumber1(num1);
        setNumber2(num2);
        setCaptchaQuestion(`What is ${num1} + ${num2}?`);
        setUserAnswer(""); // Reset user answer
        setIsCorrect(null); // Reset correctness status
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (parseInt(userAnswer, 10) === number1 + number2) {
            setIsCorrect(true);
            navigate('/chat', { replace: true }); // Navigate on correct answer
        } else {
            setIsCorrect(false);
            generateCaptcha(); // Generate new CAPTCHA on incorrect answer
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="captcha-container bg-white p-6 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold">Please complete the CAPTCHA</h2>
                <form onSubmit={handleSubmit}>
                    <label>
                        {captchaQuestion}
                        <input
                            type="text"
                            value={userAnswer}
                            onChange={(e) => setUserAnswer(e.target.value)}
                            required
                            className="border-2 rounded-md p-2 mt-2 block w-full"
                        />
                    </label>
                    <button type="submit" className="bg-customGreen text-white py-1 px-2 rounded-md mt-4 w-full">
                        Submit
                    </button>
                </form>
                {isCorrect !== null && (
                    <div className={isCorrect ? "text-green-600" : "text-red-600"}>
                        {isCorrect ? "Correct! Redirecting..." : "Incorrect, please try again."}
                    </div>
                )}
            </div>
        </div>
    );
}
