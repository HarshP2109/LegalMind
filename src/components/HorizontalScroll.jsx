import React from 'react';

export default function HorizontalScroll() {
    return (
        <div className="relative w-full overflow-hidden  py-2 -px-8 bg-customGreen ">
            <div className="flex animate-scroll whitespace-nowrap space-x-8  "> 
                <ul className="flex items-center space-x-8 text-sm bg-customGreen text-white">
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                    <li>FREE CONSULTATION</li>
                </ul>
                <ul className="flex items-center space-x-8  bg-customGreen text-white">
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>
                <li>FREE CONSULTATION</li>

                </ul>
            </div>
        </div>
    );
}
