import React from 'react'
import { AiFillInstagram } from "react-icons/ai";
import { FaYoutube } from "react-icons/fa";
import { FaFacebookSquare } from "react-icons/fa";




export default function Footer() {
    return (
        <footer className="bg-customDarkGreen p-10">

            <div
                className=' flex justify-between   items-center '>

                <div>
                    <div>
                        <img
                            src="/images/lmLogo.png"
                            className='bg-transparent h-16 w-16' />
                    </div>
                    <div className="mt-4">
                        <p><span className="text-4xl text-white">LEGAL</span><span className="text-4xl font-bold text-customGreen">MIND</span></p>
                    </div>
                    <div
                        className='mt-4'>
                        <p className="tex-normal font-thin text-white w-3/4 opacity-90">
                            Fulfill your legal need with the help
                            of LegalMind
                        </p>
                    </div>
                </div>
                <div className="gap-x-10 hidden sm:flex">
                    <div className="opacity-90">
                        <h4 className="text-white text-sm ">COMPANY</h4>
                        <ul className="text-gray-400 text-xs flex gap-y-4 mt-4 flex-col">
                            <li>Blog</li>
                            <li>Careers</li>
                            <li>Pricing</li>
                        </ul>
                    </div>
                    <div className="opacity-90">
                        <h4 className="text-white text-sm ">RESOURCES</h4>
                        <ul className="text-gray-400 text-xs flex gap-y-4 mt-4 flex-col ">
                            <li>Documentation</li>
                            <li>Papers</li>
                            <li>Press Conferences</li>
                        </ul>
                    </div>
                    <div className="opacity-90">
                        <h4 className="text-white text-sm ">LEGAL</h4>
                        <ul className="text-gray-400 text-xs  flex gap-y-4 mt-4 flex-col">
                            <li>Terms Of Services</li>
                            <li>Privacy Policy</li>
                            <li>Cookies Policy</li>
                            <li>Data Processing</li>
                        </ul>
                    </div>
                </div>
                <div>

                </div>

            </div>
            <div>

                <hr class="h-px my-8 bg-gray-200 border-0 dark:bg-gray-700" />
            </div>

            <div className="flex  items-center justify-between">
                <div>
                    <span className='text-xs text-gray-400'>© 2025 LegalMind Inc. All rights reserved</span>
                </div>
                <div>
                    <ul className="text-gray-400 flex space-x-3">
                        <li><AiFillInstagram/></li>
                        <li><FaYoutube/></li>
                        <li><FaFacebookSquare/></li>
                    </ul>
                </div>
            </div>
        </footer>
    )
}
