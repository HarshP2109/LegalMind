import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export default function SignIn() {
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        rememberMe: false
    });

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData((prevData) => ({
            ...prevData,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log('Login Data Submitted:', formData);
    };

    return (
        <div
            className='grid md:grid-cols-2  overflow-hidden'>
            <div
                className="h-screen hidden bg-customGreen items-center md:flex justify-center">
                <img src="/images/signin.png" alt="" />
            </div>
            <div className="flex justify-center items-center pt-6 ">
                <div className="w-full max-w-md ">
                    <div className="my-4 flex items-center justify-start">
                        <span className='text-[45px]'>LegalMind</span>
                        <img src="/images/lmLogo.png" className="h-12 w-12 mb-2" alt="" />
                    </div>
                    <h2 className="text-2xl font-bold ">Account Login</h2>
                    <p className="text-gray-600 mb-6">
                        If you are already a member you can login with your email address and password.
                    </p>
                    <form onSubmit={handleSubmit} className="space-y-4">
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
                                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                                placeholder="sampleuser@email.com"
                                required
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
                                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 text-gray-900 focus:outline-none sm:text-sm"
                                placeholder="••••••••"
                                required
                            />
                        </div>

                        {/* Remember Me */}
                        <div className="flex items-center">
                            <input
                                type="checkbox"
                                name="rememberMe"
                                checked={formData.rememberMe}
                                onChange={handleChange}
                                className="h-4 w-4 text-customGreen border-gray-300 rounded"
                            />
                            <label htmlFor="rememberMe" className="ml-2 block text-sm text-gray-900">
                                Remember me
                            </label>
                        </div>

                        {/* Submit Button */}
                        <div>
                            <button
                                type="submit"
                                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-customDarkGreen hover:bg-customGreen focus:outline-none transition-all"
                            >
                                Login
                            </button>
                        </div>
                    </form>

                    {/* Sign Up Link */}
                    <p className="mt-4 text-center text-sm text-gray-500">
                        Don't have an account?{' '}
                        <Link to="/signup" className="text-customGreen hover:text-customDarkGreen">
                            Sign up here
                        </Link>

                    </p>
                </div>
            </div>


        </div>
    )
}
