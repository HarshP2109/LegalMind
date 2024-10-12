import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

export default function SignUp() {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    captchaInput: ''
  });

  const [captcha, setCaptcha] = useState({ num1: 0, num2: 0 });
  const [captchaVerified, setCaptchaVerified] = useState(false);

  const generateCaptcha = () => {
    const num1 = Math.floor(Math.random() * 10); // Random number between 0-9
    const num2 = Math.floor(Math.random() * 10); // Random number between 0-9
    setCaptcha({ num1, num2 });
  };

  useEffect(() => {
    generateCaptcha(); 
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleCaptchaChange = (e) => {
    const userAnswer = parseInt(e.target.value, 10);
    const correctAnswer = captcha.num1 + captcha.num2;
    setFormData((prevData) => ({
      ...prevData,
      captchaInput: e.target.value
    }));
    setCaptchaVerified(userAnswer === correctAnswer);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      alert('Passwords do not match!');
      return;
    }
    if (!captchaVerified) {
      alert('Please solve the CAPTCHA correctly!');
      return;
    }
    console.log('Form Data Submitted:', formData);
  };

  return (
    <div className="grid md:grid-cols-2 overflow-hidden ">
      <div className="h-screen hidden bg-customGreen items-center md:flex justify-center">
        <img src="/images/signup.png" alt="" />
      </div>
      <div className=" md:px-20 px-8 py-4">
        <div>
          <div className=" flex items-center justify-start">
            <span className="text-[40px]">LegalMind</span>
            <img src="/images/lmLogo.png" className="h-12 w-12 mb-2" alt="" />
          </div>
          <h1 className="font-semibold text-2xl">Account Signup</h1>
          <p className="text-sm text-slate-600">
            Become a member and enjoy exclusive promotions.
          </p>
        </div>

        <div className="mt-2">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Full Name */}
            <div>
              <label htmlFor="fullName" className="block text-sm font-medium text-gray-700">
                Full Name
              </label>
              <input
                type="text"
                name="fullName"
                value={formData.fullName}
                onChange={handleChange}
                className=" block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                placeholder="Name"
              />
            </div>

            {/* Email Address */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email Address
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className=" block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                placeholder="Email"
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className=" block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                placeholder="••••••••"
              />
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                Confirm Password
              </label>
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                className=" block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                placeholder="••••••••"
              />
            </div>

            {/* Custom CAPTCHA */}
            <div>
              <label htmlFor="captcha" className="block text-sm font-medium text-gray-700">
                Solve this CAPTCHA: {captcha.num1} + {captcha.num2} =
              </label>
              <input
                type="number"
                name="captchaInput"
                value={formData.captchaInput}
                onChange={handleCaptchaChange}
                className=" block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                placeholder="Enter your answer"
              />
            </div>

            {/* Submit Button */}
            <div>
              <button
                type="submit"
                className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                  captchaVerified ? 'bg-customDarkGreen hover:bg-customGreen' : 'bg-gray-400'
                } focus:outline-none transition-all`}
                disabled={!captchaVerified}
              >
                Submit
              </button>
              <p className="mt-1 text-center text-sm text-gray-500">
                Already a user?{' '}
                <Link to="/signin" className="text-customGreen hover:text-customDarkGreen">
                  Sign in here
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
