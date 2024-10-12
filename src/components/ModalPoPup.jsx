import React from "react";
import { Dialog } from 'primereact/dialog';
import { Link } from "react-router-dom";

export default function ModalPoPUp({ isOpen, closeModal }) {

    return (
        <div className="card flex justify-content-center ">
            <Dialog
                header=""
                visible={isOpen}
                className="bg-customDarkGreen p-4 text-white rounded-lg"
                style={{ width: '90%', maxWidth: '400px' }}
                onHide={closeModal}  // Use closeModal directly to handle closing the modal
            ><div
                className="flex flex-col items-center justify-center">

                    <h1
                        className="text-2xl font-semibold text-customGreen">Welcome</h1>
                    <p className="m-0 text-center">
                        Log in or sign up to get smarter responses, upload files and images, and more.
                    </p>
                    <div
                        className="flex  flex-col mt-4">
                        <div
                            className="flex flex-col space-y-4">
                            <Link to="signin" className="bg-customGreen text-center text-white py-1 px-2 rounded-md hover:scale-105 transition-all inline-block">
                                Log In
                            </Link>
                            <Link to="signup" className="bg-white py-1 px-2 text-center text-customDarkGreen rounded-md hover:scale-105 transition-all inline-block">
                                Sign Up
                            </Link>
                        </div>
                        <Link to="/captcha"
                            className="text-gray-500 text-xs font-bold mt-2">
                            Stay anonymous!
                        </Link>
                    </div>
                </div>
            </Dialog>
        </div>
    );
}
